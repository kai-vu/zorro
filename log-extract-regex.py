#!/usr/bin/env python
# coding: utf-8

# # Knowledge Graphs Workshop: Pattern exercise
# 
# First, we'll load the dataset and do some basic pre-processing.
# Then we'll show the table in a handy interface.

# In[1]:


import pandas as pd
import numpy as np
from itables import show
df = pd.read_csv('Aircraft_Annotation_DataFile.csv')
df.columns = [c.lower() for c in df.columns]
df['problem'] = df['problem'].str.strip('.').str.strip()
df['action'] = df['action'].str.strip('.').str.strip()
show(df)


# ## Define a pattern for Problem strings
# 
# In this example pattern, we extract the location of the problem, the part, and a problem keyword.
# 
# Try to modify the pattern to extract more problem types, or make a different kind of problem pattern!

# In[28]:


loc_pat = '(?:(?:L|R)/H ?(?:REAR )?(?:AFT )?(?:ENG(?:INE)?,? ?)?)?(?:CYL(?:INDER)? ?)?(?:ALL )?(?:#?\d(?: ?. \d)*)?(?: CYL(?:INDER)? ?)?'
problem_pat = (
    '^'
    # The location often mentions the engine side and cylinder number
   f'(?P<loc1>{loc_pat})? ?'
    
    # A part name ends with a letter, ignore the last "S" (for plural words)
    '(?P<part>\w[ \w]+?)S?,? ' 

    f'(?:ON )?(?P<loc2>{loc_pat})? ?'

    # Match the verb but don't extract it
    '(?:IS |ARE |HAS |HAVE |APPEARS TO BE )?(?:A )?(?:POSSIBLE )?(?:EVIDENCE OF )?(?:COMING )?(?:SEVERAL )?(?:SHOWS SIGNS OF )?'

    # Some pre-defined problem keywords to match
    '(?P<problem>(?:OIL )?LEAK(?:ING)?S?|LOOSE|TORN|CRACKED|BROKEN|DAMAGED?|WORN|MISSING|BAD|SHEAR(?:ED)?|BROKE|STUCK|STICK(?:ING)?|DIRTY|DEAD|FAILED|NEEDS?|.*COMPRESSION.*)'

    f',? ?(?:ON )?(?P<loc3>{loc_pat})? ?'
    
    '(?P<rest>.*)'
)
problem_extractions = pe = df['problem'].str.extract(problem_pat)
problem_extractions['part'].replace({'HA': '', 'INDER':'', 'I':'', 'INE': '', 'ON':''}, inplace=True)
def join_cols(row):
    return ' '.join(row[~row.isna()].values.astype(str)) or np.nan
pe['location'] = pe[['loc1', 'loc2', 'loc3']].apply(join_cols, axis=1)
pe.drop(columns=['loc1', 'loc2', 'loc3'], inplace=True)

for r in ['CYLINDER', 'ENGINE', 'CYL', 'ENG', '#', ',', '&', '(', ')']:
  pe['location'] = pe['location'].str.replace(r, '')
pe['location'] = pe['location'].str.replace('ALL', '1 2 3 4')
pe['cylinders'] = pe['location'].str.extractall('(\d)')[0].groupby(level=0).apply(
   lambda x: ' '.join(set(x))
)
pe['engine'] = pe['location'].str.replace('\d', '', regex=True).str.strip()
pe.drop(columns=['location'], inplace=True)

# Fix compression
comp = (~pe['problem'].isna() & pe['problem'].str.contains('COMPRESSION'))
pe.loc[comp, 'rest'] = pe[comp]['problem'] + pe[comp]['rest']
pe.loc[comp, 'problem'] = 'COMPRESSION'
pe.insert(0, 'id', df['ident'])

# Filter parts
ts = pd.read_csv('prompt-extracted/part-classes.tsv', sep='\t')['Part']
ptc = pe['part'].dropna().apply(lambda x: x.split()[-1] if x else None).value_counts()
ok_parts = set(ptc[ptc.index.difference(ts)].sort_values().tail(10).index) | set(ts)
filter_parts = lambda x: x if x and x.split()[-1] in ok_parts else None
pe['part'] = pe['part'].fillna('').apply(filter_parts)

pe.to_csv('problem_extractions_regex.csv', index=None)


# Show the most common problem extractions
show(problem_extractions.fillna('').value_counts())


# In[5]:


pc = pe['part'].value_counts().drop(['ENGINE'])
part_pat = '|'.join(pc[(pc.index.str.len()>3) & (pc>=3)].index)
locs = pe['location'].str.replace(' +', ' ', regex=True).str.strip().unique()
loc_pat = '|'.join(locs[pd.Series(locs).str.len()>1])
df_nopart_noloc = df.loc[(df['problem'].str.count(part_pat)==0) & (df['problem'].str.count(loc_pat)==0)]
df_nopart_noloc['problem'].to_csv('problems_nopart_noloc.csv', index=None)
show(df_nopart_noloc)


# In[4]:


# Show non-matching problems
problems_nomatch = df['problem'].loc[problem_extractions.isna().all(axis=1)]
problems_nomatch.to_csv('problems_nomatch.csv', index=None)
show(problems_nomatch.value_counts().rename('count'))


# ## Define a pattern for Action strings
# 
# In this example pattern, we extract the location of the action, the part, and an action keyword.
# 
# Try to modify the pattern to extract more action types, or make a different kind of action pattern!

# In[36]:


action_pat = (
    '^(?:REMOVED & )?(?:RE)?'
    # Pre-defined action keywords
    '(?P<action>REPLACED|TIGHTENED|SECURED|ATTACHED|FASTENED|TORQUED|CLEANED|STOP DRILLED) ?'

    # The location often mentions the engine side and cylinder number
    '(?P<location>(?:(?:L|R)/H (?:ENG )?)?(?:CYL ?)?(?:#?\d(?: ?. \d)*)(?: CYL ?)?)? ?'

    # Often, replacements mention "W/ NEW"; ignore it
    '(?:W/ )?(?:NEW )?'

    # A part name ends with a letter, ignore the last "S" (for plural words)
    '(?P<part>[^,.]*?\w)S?'
    
    '(?: W/ .*)?(?:[,.] .*)?$'
)
action_extractions = ae = df['action'].str.extract(action_pat)
ae.insert(0, 'id', df['ident'])

for r in ['CYLINDER', 'ENGINE', 'CYL', 'ENG', '#', ',', '&', '(', ')']:
  ae['location'] = ae['location'].str.replace(r, '')
ae['location'] = ae['location'].str.replace('ALL', '1 2 3 4')
ae['cylinders'] = ae['location'].str.extractall('(\d)')[0].groupby(level=0).apply(
   lambda x: ' '.join(set(x))
)
ae['engine'] = ae['location'].str.replace('\d', '', regex=True).str.strip()
ae.drop(columns=['location'], inplace=True)


# Filter parts
ts = pd.read_csv('prompt-extracted/part-classes.tsv', sep='\t')['Part']
ptc = ae['part'].dropna().apply(lambda x: x.split()[-1] if x else None).value_counts()
ok_parts = set(ptc[ptc.index.difference(ts)].sort_values().tail(10).index) | set(ts)
filter_parts = lambda x: x if x and x.split()[-1] in ok_parts else None
ae['part'] = ae['part'].fillna('').apply(filter_parts)

action_extractions.to_csv('action_extractions_regex.csv', index=None)

show(action_extractions.fillna('').value_counts())


# In[33]:


# Show non-matching actions
show(df['action'].loc[action_extractions.isna().all(axis=1)])


# ## Loading extractions into graph
# 
# Now, we'll transform our extractions into graphs and load them into the Knowledge Graph.

# In[ ]:


from helperFunctions import obj_to_triples
import re
from rdflib import Graph, URIRef, BNode, Literal, RDF, RDFS, DC, Namespace
ZORRO = Namespace("https://zorro-project.nl/example/")

def create_problem_obj(row):
    ent = ZORRO[f'problem{row.ident}']
    
    problem_match = re.search(problem_pat, row.problem)
    problem_fields = problem_match.groupdict() if problem_match else {}
    action_match = re.search(action_pat, row.action)
    action_fields = action_match.groupdict() if action_match else {}

    def camelcase(fields, name):
        # Convert string into a clean CamelCase name
        return re.subn('\W', '', fields.get(name, '').title())[0]
    
    return {
        '@id': ent,
        RDF.type: ZORRO[camelcase(problem_fields, 'problem') + 'Problem'],
        DC.description: Literal(row.problem),
        
        ZORRO.involvedPart: {
            RDF.type: ZORRO[camelcase(problem_fields, 'part') + 'Part'],
            ZORRO.location: Literal((problem_fields.get('location') or '').strip())
        } if problem_fields.get('part') else None,
        
        ZORRO.requiredAction: {
            DC.description: Literal(row.action),
            RDF.type: ZORRO[camelcase(action_fields, 'action') + 'Action'],
            
            ZORRO.involvedPart: {
                RDF.type: ZORRO[camelcase(action_fields, 'part') + 'Part'],
                ZORRO.location: Literal((action_fields.get('location') or '').strip())
            } if action_fields.get('part') else None
        }
    }

# Show the turtle serialization of the first 5 extractions
g = Graph()
g.namespace_manager.bind('', ZORRO)
for obj in df.head(5).apply(create_problem_obj, axis=1):
    for t in obj_to_triples(obj):
        g.add(t)
print(g.serialize())


# In[ ]:


# Run on entire dataset, takes a few seconds!
g = Graph()
g.namespace_manager.bind('', zorro)
for obj in df.apply(create_problem_obj, axis=1):
    for t in obj_to_triples(obj):
        g.add(t)
g.serialize('pattern_graph.ttl')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'ipython_sparql_pandas')
from helperFunctions import GraphDB

db = GraphDB()
repo_name = 'zorro'
db.create_repo(repo_name).text

response = db.load_data(repo_name, 'pattern_graph.ttl', 
          graph_name = "https://zorro-project.nl/example/PatternGraph")
print(response.text)


# In[ ]:


get_ipython().run_cell_magic('sparql', 'http://localhost:{db.port}/repositories/{repo_name} -s result', 'PREFIX : <https://zorro-project.nl/example/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\nselect (count(*) as ?c) where { \n\t?prob a :Problem .\n}\n')


# ### But didn't we add many more instances ..?
# 
# These are only the ones for which we couldn't extract a more specific problem class!
# 
# To get our instances, we also load our schema about maintenance:

# In[ ]:


response = db.load_data(repo_name, 'maintenance.ttl', 
          graph_name = "https://zorro-project.nl/example/MaintenanceGraph")
print(response.text)


# In[ ]:


get_ipython().run_cell_magic('sparql', 'http://localhost:{db.port}/repositories/{repo_name} -s result', 'PREFIX : <https://zorro-project.nl/example/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\nselect (count(*) as ?c) where { \n\t ?prob a :Problem .\n}\n')


# In[ ]:


get_ipython().run_cell_magic('sparql', 'http://localhost:{db.port}/repositories/{repo_name} -s result', 'PREFIX : <https://zorro-project.nl/example/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\nselect ?problemClass (count(*) as ?count) where { \n    ?prob a :Problem .\n    ?prob a ?problemClass . \n}\nGROUP BY ?problemClass\n')


# In[ ]:


result.set_index('problemClass')['count'].plot.barh()


# In[ ]:




