#!/usr/bin/env python
# coding: utf-8

# # PDF and Prompt extracted tables

# In[5]:


import csv, rdflib, re

g = rdflib.Graph()
base = rdflib.Namespace('https://w3id.org/zorro#')
g.bind(None, base)
troubleshooting = csv.DictReader(open('pdf-extracted/troubleshooting.csv'))
for line in troubleshooting:
    trouble, cause, remedy = [
      base[(re.sub('[^a-zA-z]', '', ' '.join(line[key].split()[:10]).title()))]
      for key in ['TROUBLE', 'PROBABLE CAUSE', 'REMEDY']
    ]

    triples = [
        (trouble, rdflib.RDFS['subClassOf'], base['Problem']),
        (trouble, rdflib.RDFS['label'], rdflib.Literal(line['TROUBLE'])),
        
        (cause, rdflib.RDFS['subClassOf'], base['Problem']),
        (cause, rdflib.RDFS['label'], rdflib.Literal(line['PROBABLE CAUSE'])),
        (trouble, base['hasCause'], cause),
        
        (remedy, rdflib.RDFS['subClassOf'], base['Solution']),
        (remedy, rdflib.RDFS['label'], rdflib.Literal(line['REMEDY'])),
        (remedy, base["solves"], cause),
    ]
    for t in triples:
        g.add(t)

g.serialize('generated-rdf/troubleshooting.ttl', format='ttl')


# In[ ]:


import csv, rdflib, re

g = rdflib.Graph()
base = rdflib.Namespace('https://w3id.org/zorro#')
g.bind(None, base)

def ensure_superclass_chain(part, cls):
    if cls not in part_class_lookup:
        g.add((cls, rdflib.RDFS['subClassOf'], base['Part']))
    else:
        parent_cls = part_class_lookup[cls]
        g.add((cls, rdflib.RDFS['subClassOf'], parent_cls))
        ensure_superclass_chain(cls, parent_cls)

part_class_lookup = {}
lines = csv.DictReader(open('part-classes.tsv'), delimiter='\t')
for line in lines:
    part, cls = [
        base[re.sub('[^a-zA-Z]', '', line[key].split('(')[0].title())]
        for key in ['Part', 'subClassOf']
    ]

    triples = [
        (part, rdflib.RDFS['label'], rdflib.Literal(line['Part'].title())),
        (part, rdflib.RDFS['subClassOf'], cls),
        (cls, rdflib.RDFS['label'], rdflib.Literal(line['subClassOf'].title())),
    ]
    for t in triples:
        g.add(t)
    part_class_lookup[part] = cls

    ensure_superclass_chain(part, cls)

parts_catalog = csv.DictReader(open('parts-catalog.csv'))
for line in parts_catalog:

    system, assembly = [
        base[re.sub('[^a-zA-Z]', '', line[key].title())]
        for key in ['Section', 'Figure']
    ]

    cls = base[re.sub('[^a-zA-Z]', '', line['Type'].title())]

    label = line['Specifics'].strip() + ' ' + re.sub('[^a-zA-Z]', '', line['Type'].title())

    s = base['partnr-' + line['Part Number']]

    if cls in part_class_lookup:
        part_type_class = cls
    else:
        part_type_class = base['Part']

    triples = [
        (s, rdflib.RDFS['subClassOf'], cls),
        (s, base['partOf'], assembly),
        (s, base['partOf'], system),
        (s, base['partNumber'], rdflib.Literal(line['Part Number'])),
        (s, rdflib.RDFS['label'], rdflib.Literal(label)),

        (assembly, rdflib.RDFS['subClassOf'], base['Assembly']),
        (assembly, rdflib.RDFS['label'], rdflib.Literal(line['Figure'])),

        (system, rdflib.RDFS['subClassOf'], base['System']),
        (system, rdflib.RDFS['label'], rdflib.Literal(line['Section'])),
    ]
    for t in triples:
        g.add(t)

    # If the type (cls) is not in the lookup, define it as a subclass of "Part"
    if cls not in part_class_lookup:
        g.add((cls, rdflib.RDFS['subClassOf'], base['Part']))
        g.add((cls, rdflib.RDFS['label'], rdflib.Literal(re.sub('[^a-zA-Z]', '', line['Type'].title()))))
    else:

        ensure_superclass_chain(cls, part_class_lookup[cls])

g.serialize('generated-rdf/part-catalog.ttl', format='ttl')


# In[ ]:


import csv, rdflib, re

g = rdflib.Graph()
base = rdflib.Namespace('https://w3id.org/zorro#')
g.bind(None, base)
lines = csv.DictReader(
    open('prompt-extracted/problem-component-function.tsv'),
    delimiter='\t'
)
for line in lines:
    problem, component, function = [
        base[ re.sub('[^a-zA-Z]', '', line[key].split('(')[0].title()) ]
        for key in ['defines','functionOf','Function']
    ]

    triples = [
        (function, rdflib.RDFS['subClassOf'], base['Function']),
        (function, rdflib.RDFS['label'], rdflib.Literal(line['Function'])),
        (function, base['defines'], problem),

        (problem, rdflib.RDFS['label'], rdflib.Literal(line['defines'])),

        (component, rdflib.RDFS['subClassOf'], base['Component']),
        (component, rdflib.RDFS['label'], rdflib.Literal(line['functionOf'])),
        (component, base['hasFunction'], function),
    ]
    for t in triples:
        g.add(t)


lines = csv.DictReader(
    open('prompt-extracted/functions.tsv'),
    delimiter='\t'
)
for line in lines:
    component, function = [
        base[ re.sub('[^a-zA-Z]', '', line[key].split('(')[0].title()) ]
        for key in ['Component','hasFunction']
    ]

    triples = [
        (function, rdflib.RDFS['label'], rdflib.Literal(line['hasFunction'])),
        (function, rdflib.RDFS['subClassOf'], base['Function']),
        (component, base['hasFunction'], function),
        (component, rdflib.RDFS['subClassOf'], base['Component']),
    ]
     for t in triples:
        g.add(t)


lines = csv.DictReader(
    open('prompt-extracted/subfunction.tsv'),
    delimiter='\t'
)
for line in lines:
    function, subfunction = [
        base[ re.sub('[^a-zA-Z]', '', line[key].split('(')[0].title()) ]
        for key in ['subFunctionOf','Function']
    ]

    triples = [
        (subfunction, base['subFunctionOf'], function),
    ]
    for t in triples:
        g.add(t)

lines = csv.DictReader(
    open('prompt-extracted/dependsOn.tsv'),
    delimiter='\t'
)
for line in lines:
    c1, c2 = [
        base[ re.sub('[^a-zA-Z]', '', line[key].split('(')[0].title()) ]
        for key in ['Component','dependsOn']
    ]

    triples = [
        (c1, base['dependsOn'], c2),
    ]
    for t in triples:
        g.add(t)


g.serialize('generated-rdf/functions.ttl', format='ttl')


# # Maintenance logbook extraction tables

# In[2]:


import csv, json, rdflib, re, tqdm
a = rdflib.RDF['type']
subclassof = rdflib.RDFS['subClassOf']
label = rdflib.RDFS['label']

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemma(v):
  return lemmatizer.lemmatize(v.lower(), pos='v').upper()

event_logs = csv.DictReader(
    open('Aircraft_Annotation_DataFile.csv', encoding='utf-8-sig'),
)
event_logs = { e['IDENT']:e for e in event_logs }

def make_event(g, line, event):
  log = event_logs[line['id']]
  situation = base[event.lower() + str(log['IDENT'])]
  g.add( (situation, label, rdflib.Literal(event.lower() + str(log['IDENT']))) )
  g.add( (situation, rdflib.DC['description'], rdflib.Literal(log[event.upper()])) )

  # id,part,situation,engine,cyl
  if line[event]:
    event_label = line[event].title() + ' ' + event.title()
    event_type = base[event_label.replace(' ','')]
    g.add((situation, a, event_type ))
    g.add((event_type, subclassof, base[event.title()] ))
    g.add((event_type, label, rdflib.Literal(event_label) ))

  cyls = re.findall('\d', line['cylinders']) if line['cylinders'] else [None]
  for c in cyls:
    if line['part']:
      part_name = re.sub('(S$|[^\s\w])', '', line['part']).lower()
      part = rdflib.BNode()
      g.add((situation, base['involves'], part ))

      parttype = base[''.join(part_name.title().split())]
      g.add((part, a, parttype))
      g.add((parttype, label, rdflib.Literal(part_name) ))
      partclass = base[part_name.split()[-1].title()]
      g.add((parttype, subclassof, partclass))
      g.add((partclass, subclassof, base['Part']))

      if line['engine']:
        g.add((part, base['atEngine'], rdflib.Literal(line['engine']) ))
      if c:
        g.add((part, base['atCylinder'], rdflib.Literal(int(c)) ))
  return situation

for source in ['regex', 'chatgpt_4o']:

  g = rdflib.Graph()
  base = rdflib.Namespace('https://w3id.org/zorro#')
  g.bind(None, base)

  fname = f'log-extracted/problem_extractions_{source}.csv'
  for line in tqdm.tqdm(csv.DictReader(open(fname)), desc=fname):
    make_event(g, line, 'problem')

  fname = f'log-extracted/action_extractions_{source}.csv'
  for line in tqdm.tqdm(csv.DictReader(open(fname)), desc=fname):
    problem = base['problem' + str(line['id'])]
    action = make_event(g, line, 'action')
    g.add( (action, base['dealsWith'], problem) )


  g.serialize(f'generated-rdf/extractions_{source}.ttl', format='ttl')


# In[ ]:


g = rdflib.Graph()
base = rdflib.Namespace('https://w3id.org/zorro#')
g.bind(None, base)

for row in open('part-links/part-links-regex.tsv'):
  part_name, part_scores = row.split('\t')
  part_uri = base[re.sub('(S$|\W)', '', part_name.title())]
  for candidate, score in json.loads(part_scores).items():
    link = rdflib.BNode()
    g.add((part_uri, base['isMaybe'], link))
    g.add((link, base['linkCandidate'], base[candidate]))
    g.add((link, base['linkScore'], rdflib.Literal(score)))

g.serialize(f'generated-rdf/part-links-regex.ttl', format='ttl')


# In[5]:


import csv, json, rdflib, re, collections

g = rdflib.Graph()
base = rdflib.Namespace('https://w3id.org/zorro#')
g.bind(None, base)

for row in open('part-links.tsv'):
  part_name, part_scores = row.split('\t')
  part_uri = base[re.sub('(S$|\W)', '', part_name.title())]
  candidates = collections.Counter(json.loads(part_scores))
  
  for candidate, score in candidates.most_common():
    g.add((part_uri, base['isProbably'], base[candidate]))
    break

g.serialize(f'generated-rdf/part-links-simplified.ttl', format='ttl')

