# made by Niek Aukes
# this file is a bit of a mess still.
# it's used to experiment with a part linker
#%%
import numpy as np
import pandas as pd
from itertools import chain

#%%
# load in problem extractions and parts catalog
problem_extractions = pd.read_csv('problem_extractions_chatgpt_4o.csv')
action_extractions = pd.read_csv('action_extractions_chatgpt_4o.csv')
parts_catalog = pd.read_csv('pdf-extracted/parts-catalog.csv')

#%%
# create a dictionary with the part number and type
part_dict = {}
for index, row in parts_catalog.iterrows():
    part_dict[row['Part Number']] = row['Type']
    
print(part_dict)
# %%
# print the unique part types
part_set = set(part_dict.values())
print(part_set)

# %%
# calculate the % of parts mentioned in the problem extractions 
# that are not in the parts catalog
count = 0
for index, row in problem_extractions.iterrows():
    if row['part'] not in part_set:
        count += 1
        print(row['part'])

print(count/len(problem_extractions))
print(count)

# from analyzing the output, it seems that most parts
# mentioned in the problem extractions that aren't in the parts catalog
# seem to be assemblies, like 'ENGINE',
# abbreviations, like 'CYL'
# or still have location identifiers attached to them, like in 'ROCKER COVER GASKETS'

# %%

# idea: split up the parts in problem extractions by spaces
count = 0
for index, row in problem_extractions.iterrows():
    parts = str(row['part']).split(' ')
    for part in parts:
        if part in part_set:
            count += 1
            print("part:", row['part'], "match:", part)
            break

print("% of parts mentioned in the catalog:", count/len(problem_extractions))

# much higher percentage of parts mentioned in the problem extractions
# however there is a lot of noise as well
# like part: INTAKE TUBE GASKET match: TUBE
# this is where a more structured approach could be useful, like a parser
# %%


# =============================================================================
#   LEXING AND PARSING IDEA
# =============================================================================
# for this idea, we make use of concepts from compiler design 
# to structurally analyze the problem extractions

# 1. Lexing: split the problem extractions into tokens
# TOKENS: PART, ATTRIBUTE, CONTEXT
# all words are classified as either a part, an attribute, or context
# CONTEXT is a word that doesn't fit as either a PART or ATTRIBUTE

TOKENS = ['PART', 'ATTRIBUTE', 'CONTEXT']
ATTRIBUTES = [
    "BLACK",
    "BOX",
    "ANGLE",
    "INTERCONNECT",
    "MOUNTING",
    "BLAST",
    "INNER",
    "SNAP",
    #"INTAKE", # actually a "part", but doesn't exist in the parts catalog
    "PUSH",
    "BOTTOM",
    "WIRE",
    "BACK",
    "INDUCTION"
]
wordmap = {
    #"ENGINE": 'PART', # ENGINE would match with all parts, practically useless
    "INTAKE": 'PART', # a subset of parts, but we treat it as a part, even though we can't find it in the parts catalog
}
for part in part_set:
    wordmap[part] = 'PART'
    wordmap[part + "S"] = 'PART'
    
    # if the part is an assembly, remove the last word
    splt = part.split(' ')
    if len(splt) > 1 and splt[-1] == 'ASSEMBLY':
        wordmap[' '.join(splt[:-1])] = 'PART'
    
for attribute in ATTRIBUTES:
    if attribute in wordmap:
        print("WARNING: attribute already in wordmap")
    wordmap[attribute] = 'ATTRIBUTE'
    
test_sentence = "ROCKER COVER GASKET"

def lex(sentence):
    tokens = []
    words = str(sentence).split(' ')
    for word in words:
        if word in wordmap:
            tokens.append((word, wordmap[word]))
        else:
            tokens.append((word, 'CONTEXT'))
    return tokens

print(lex(test_sentence))
#%%
# 2. Parsing: create a tree structure from the tokens
# tree structure:
# main: ctx part ctx
# part: ATTRIBUTE part
#     | PART part
#     | -
# ctx : CONTEXT ctx
#     | -
# we don't care about ambiguity in the parsing,
# as examples are small enough to brute force the parsing

class ParseNode:
    def __init__(self, type, value, children):
        self.type = type
        self.value = value
        self.children = children
        
    def __str__(self):
        s = self.type
        if self.value != '':
            s += ' ' + self.value
        s += ' ('
        for child in self.children:
            s += str(child) + ', '
        s += ')'
        return s
    
    def __repr__(self) -> str:
        return self.__str__()
    
class UnparsableException(Exception):
    pass

def parse(tokens):
    try:
        ctx1 = parse_ctx(tokens)
        part = parse_part(tokens, True)
        ctx2 = parse_ctx(tokens)
        if tokens != []:
            raise UnparsableException
        return ParseNode('main', '', [ctx1, part, ctx2])
    except UnparsableException:
        return None

def parse_part(tokens, force=False):
    if tokens == []:
        if force:
            raise UnparsableException
        return ParseNode("", "", [])
    
    next = tokens.pop(0)
    if next[1] == 'CONTEXT':
        tokens.insert(0, next)
        if force:
            raise UnparsableException
        return ParseNode("part", "", [])
    if next[1] == 'ATTRIBUTE':
        leaf = ParseNode('ATTRIBUTE', next[0], [])
        return ParseNode('part', "", [leaf, parse_part(tokens, force)])
    elif next[1] == 'PART':
        leaf = ParseNode('PART', next[0], [])
        return ParseNode('part', "", [leaf, parse_part(tokens, False)])
    else:
        raise UnparsableException

def parse_ctx(tokens):
    if tokens == []:
        return ParseNode("ctx", "", [])
    
    next = tokens.pop(0)
    if next[1] == 'CONTEXT':
        leaf = ParseNode('CONTEXT', next[0], [])
        return ParseNode('ctx', "", [leaf, parse_ctx(tokens)])
    else:
        tokens.insert(0, next)
        return ParseNode("ctx", "", [])
    
print(parse(lex(test_sentence)))
# %%
# check parsability of all problem extractions
parsable = 0
parsed_extractions = []
for index, row in problem_extractions.iterrows():
    result = parse(lex(row['part']))
    if result is not None:
        parsable += 1
        parsed_extractions.append(result)
    else:
        lexed = lex(row['part'])
        print("Unparsable:", 
              row['part'],
              "with tokens",
              [word[1] for word in lexed])
        
print("Parsable:", parsable/len(problem_extractions))
# %%
#==============================================================================
#   STRUCTURING THE PARTS CATALOG
#==============================================================================
# Idea: create a graph with all parts being nodes, these parts have relations
# to other parts, like being a subpart of an assembly, or being a part of the same assembly
# with this graph structure, we may be able to identify parts based on mentions of other parts

# 1. create a graph with all parts as nodes

class Part:
    def __init__(self, part_number, part_type, specifics):
        self.part_number = part_number
        self.part_type = part_type
        self.specifics = specifics
        self.connections = set()

    def __str__(self):
        return self.part_number + ' (' + self.part_type + ')'
    
    def __repr__(self) -> str:
        return self.__str__()

class Connection:
    def __init__(self, part1, part2, relation):
        self.part1 = part1
        self.part2 = part2
        self.relation = relation

    def __str__(self):
        return self.part1.part_number + ' ' + self.relation + ' ' + self.part2.part_number
    
    def __repr__(self) -> str:
        return self.__str__()
    
class Section:
    def __init__(self, name):
        self.name = name
        self.assemblies = {}
        
    def __str__(self):
        return self.name + ' (' + str(len(self.parts)) + ' parts)'
    
    def __repr__(self) -> str:
        return self.__str__()
    
class Assembly:
    def __init__(self, name):
        self.name = name
        self.parts = {}
        
    def __str__(self):
        return self.name + ' (' + str(len(self.parts)) + ' parts)'
    
    def __repr__(self) -> str:
        return self.__str__()

#%%
word_hints = {}
sections = {}
occurance = {}

for index, row in parts_catalog.iterrows():
    section = row['Section']
    assembly = row['Figure']
    if section not in sections:
        sections[section] = Section(section)
    if assembly not in sections[section].assemblies:
        sections[section].assemblies[assembly] = Assembly(assembly)
    
    part_name = str(row['Part Number'])
    
    if part_name in sections[section].assemblies[assembly].parts:
        print("WARNING: part already in assembly:", part_name)
        continue
    
    part = Part(part_name, row['Type'], row['Specifics'])
    sections[section].assemblies[assembly].parts[part.part_number] = part
    #buzzwords = str(row['Part Number']).split(' ')
    buzzwords = []
    buzzwords.extend(str(row['Specifics']).split(' '))
    buzzwords.extend(str(row['Type']).split(' '))
    buzzwords = [str(w).upper() for w in set(buzzwords)]
    for w in buzzwords:
        if w in word_hints:
            word_hints[w].append(part)
        else:
            word_hints[w] = [part]
    occurance[part] = len(buzzwords)

# build connections
for section in sections.values():
    for assembly in section.assemblies.values():
        for part in assembly.parts.values():
            for other in assembly.parts.values():
                if part != other:
                    c = Connection(part, other, 'ASSEMBLY')
                    part.connections.add(c)
                    other.connections.add(c)
print(word_hints['ROCKER'])
# %%

# example
parsed_example = parse(lex(test_sentence))
#print(parsed_example)
# find the deepest part node
def find_deepest_part(node):
    if node.type == 'main':
        return find_deepest_part(node.children[1])
    elif node.type == 'part':
        for child in node.children[::-1]: # iterate in reverse order
            if child.type == 'part':
                d = find_deepest_part(child)
                if d is not None:
                    return d
            elif child.type == 'PART':
                return child
    return None

dpart = find_deepest_part(parsed_example)

# find all buzzwords
def find_buzzwords(node, details=""):
    buzzwords = []
    if node.type == 'main':
        for child in node.children:
            buzzwords.extend(find_buzzwords(child))
    elif node.type == 'ctx':
        for child in node.children:
            buzzwords.extend(find_buzzwords(child))
    elif node.type == 'part':
        for child in node.children:
            buzzwords.extend(find_buzzwords(child))
    elif node.type == 'ATTRIBUTE':
        buzzwords.append(node.value)
    elif node.type == 'PART':
        buzzwords.append(node.value)
        if node.value.endswith('S'):
            buzzwords.append(node.value[:-1]) # remove the S
    elif node.type == 'CONTEXT':
        buzzwords.append(node.value)
    details_split = details.split(' ')
    details_split = [d.upper() for d in details_split if d != '']
    buzzwords.extend(details_split)
    
    return buzzwords
            
bz = find_buzzwords(parsed_example)
print(bz)

def make_buzz_ranking(buzzwords):
    ranking = {}
    for buzzword in buzzwords:
        if buzzword in word_hints:
            for part in word_hints[buzzword]:
                if part in ranking:
                    ranking[part] += 100.0 / len(set(chain(word_hints[buzzword],buzzwords)))
                else:
                    ranking[part] = 100.0 / len(set(chain(word_hints[buzzword],buzzwords)))
    return ranking

def inv(d):
    r = {}
    for k, v in d.items():
        if v in r:
            r[v].append(k)
        else:
            r[v] = [k]
    return r
ranking_example = make_buzz_ranking(bz)
print(ranking_example)
print(len(ranking_example) / len(parts_catalog))

# %%
# filter out the parts that don't match the type
def get_candidates(ranking, part_type):
    candidates = []
    for part, score in ranking.items():
        if (part.part_type == part_type 
        or part.part_type == part_type + ' ASSEMBLY'
        or part.part_type + 'S' == part_type):
            candidates.append((part, score))
    return candidates

example_candidates = get_candidates(ranking_example, dpart.value)
print(example_candidates)

def rank_candidates(ranking, candidates, ignore_type=None):
    scores = {}
    for candidate in candidates:
        score = 1
        for connection in candidate[0].connections:
            subjectpart = connection.part1 if connection.part1 != candidate[0] else connection.part2
            
            # ignore if the part is of the ignore_type
            # usually used for the part itself
            if ignore_type is not None and subjectpart.part_type == ignore_type:
                continue
            if subjectpart in ranking:
                score += ranking[subjectpart]
        #score *= candidate[1]
        score += candidate[1] * 5
        scores[candidate[0]] = score
    return scores

print(rank_candidates(ranking_example, example_candidates, dpart.value))
# %%
def get_ranked_candidates(node, secondary=None):
    dpart = find_deepest_part(node)
    bz = find_buzzwords(node)
    if dpart is None: # no part found
        print("No part found for", node)
        return None
    
    ranking = make_buzz_ranking(bz)
    
    # if there is a secondary part, use the ranking
    # may be used for part pairs in problem and action extractions
    if secondary is not None:
        s_ranking = make_buzz_ranking(find_buzzwords(secondary))
        # add the scores of the secondary ranking to the primary ranking
        for part, score in s_ranking.items():
            if part in ranking:
                ranking[part] += score * 0.5
            else:
                ranking[part] = score * 0.5
    
    candidates = get_candidates(ranking, dpart.value)
    if len(candidates) == 0:
        print("No candidates found for", dpart.value)
    return rank_candidates(ranking, candidates, dpart.value)

#%%
# run the function on all parsed extractions
ranked_candidates = []
for parsed in parsed_extractions:
    res = get_ranked_candidates(parsed)
    ranked_candidates.append(res)
    print("Ranked candidates:", res)

extraction_successes = 0
identified_parts = []
for c in ranked_candidates:
    if c is not None and len(c) > 0:
        extraction_successes += 1
        # get the part with the highest score
        # and only add it if it is the only part with that score
        m = max(c.values())
        invc = inv(c)
        if len(invc[m]) == 1 and m > 4:
            identified_parts.append(invc[m][0])
        else:
            print("Ambiguity in ranking:", invc[m],"with score", m)
        
print("succesful extractions:", extraction_successes / len(ranked_candidates))
print("succesful identifications:", len(identified_parts) / len(ranked_candidates))

# scrolling through the output, it seems that the parser is able to identify
# clearly mentioned parts, like seals and gaskets
# but struggles a lot with more general mentions, like 'ENGINE' or 'INTAKE'
# obviously this makes sense, there is not a single part that is called 'ENGINE' or 'INTAKE'
# but rather a collection of parts that make up the engine or intake system

# now it is unknown wether it actually identifies the correct part
# not sure how to evaluate this.
# %%
# custom example playground
examples = [
    "ROCKER COVER",
    "ROCKER COVER GASKETS",
    "BLACK BAFFLE SEAL", # example of a not so strong match
    "INTAKE GASKET"
]
for example in examples:
    example_parsed = parse(lex(example))
    ranked_candidates = get_ranked_candidates(example_parsed)
    print("Ranked candidates for", example +  ":", ranked_candidates)
    
    
    
    
    
    
    
    
# %%
# now, process the problem and action extractions together
for i in range(len(problem_extractions)):
    problem = problem_extractions.iloc[i]
    action = action_extractions.iloc[i]
    problem_parsed = parse(lex(problem['part']))
    action_parsed = parse(lex(action['part']))
    if problem_parsed is not None:
        problem_candidates = get_ranked_candidates(problem_parsed, action_parsed)
        print("Problem:", problem['part'])
        print("Problem candidates:", problem_candidates)
    
    if action_parsed is not None:
        action_candidates = get_ranked_candidates(action_parsed, problem_parsed)
        print("Action:", action['part'])
        print("Action candidates:", action_candidates)
# %%
# $$ \text{score}(entry, catalog) = \frac{|\text{entry} \cap \text{catalog}|}{|\text{entry} \cup \text{catalog}|} $$