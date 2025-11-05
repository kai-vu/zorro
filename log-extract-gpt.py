#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from itables import show
df = pd.read_csv('Aircraft_Annotation_DataFile.csv')
df.columns = [c.lower() for c in df.columns]
df['problem'] = df['problem'].str.strip('.').str.strip()
df['action'] = df['action'].str.strip('.').str.strip()
show(df)


# In[ ]:


from openai import OpenAI
client = OpenAI(api_key=open('openai-key.txt').read().strip())

prompt = """Given a set of aircraft maintenance problem descriptions, output a json object with an array of problem objects: {"problems":[ problem1, problem2, ... ]}
Each problem is an object with the following fields:
- "id" (id of problem, first number in description)
- "engine" (where the problem occurs. Single-letter string ("R" or "L") or null)
- "cylinders" (array of integers, where the problem occurs, e.g. [1]. If "all" is mentioned, output [1,2,3,4])
- "part" (which single part causes the problem, e.g. "ENGINE BAFFLE")
- "problem" (ONE single word of what is going on, e.g. "LEAK", "FELL_OFF", "CRACK", "DAMAGE", "COMPRESSION", "NEED")
- "details" (more info that doesn't fit in the other fields, e.g. "BADLY")
The values of the fields MUST be substrings (spans) in the problem description.
All fields are optional except for "id".
If there is no very clear single part identified, don't fill in the field!
"""
from tqdm import tqdm
import json, pathlib

cache = pathlib.Path('gpt-cache')
cache.mkdir(parents=True, exist_ok=True)

out = []
for i in tqdm(range(0, len(df), 20)):
  lines = df.iloc[i:i+20][['ident', 'problem']].astype(str).apply(' '.join, axis=1)

  completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    temperature= 0,
    response_format={ "type": "json_object" },
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": '\n'.join(lines)}
    ]
  )
  c = completion.choices[0].message.content
  with open(cache / f'problem-{i}.json', 'w') as fw:
    print(c, file=fw)
  out += json.loads(c).get('problems', [])
out


# In[ ]:


p = pd.DataFrame.from_records(out)
p.to_csv('log-extracted/problem_extractions_chatgpt_4o.csv', index=None)
show(p)


# In[ ]:


from openai import OpenAI
client = OpenAI(api_key=open('openai-key.txt').read().strip())

prompt = """Given a set of aircraft maintenance action descriptions, output a json object with an array of action objects: {"actions":[ action1, action2, ... ]}
Each action is an object with the following fields:
- "id" (id of action, first number in description)
- "engine" (where the action occurs. Single-letter string ("R" or "L") or null)
- "cylinders" (array of integers, where the action occurs, e.g. [1]. If "all" is mentioned, output [1,2,3,4])
- "part" (which single part causes the action, e.g. "ENGINE BAFFLE")
- "action" (ONE single word of what was done, e.g. "REPLACED", "TIGHTENED", "INSTALLED")
- "details" (more info that doesn't fit in the other fields, e.g. "NEW", "LEAK CHECK GOOD")
The values of the fields MUST be substrings (spans) in the action description.
All fields are optional except for "id".
If there is no very clear single part identified, don't fill in the field!
"""
from tqdm import tqdm
import json, pathlib

cache = pathlib.Path('gpt-cache')
cache.mkdir(parents=True, exist_ok=True)

act = []
for i in tqdm(range(0, len(df), 20)):
  lines = df.iloc[i:i+20][['ident', 'action']].astype(str).apply(' '.join, axis=1)

  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature= 0,
    response_format={ "type": "json_object" },
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": '\n'.join(lines)}
    ]
  )
  c = completion.choices[0].message.content
  with open(cache / f'action-{i}.json', 'w') as fw:
    print(c, file=fw)
  act += json.loads(c).get('actions', [])
act


# In[ ]:


a = pd.DataFrame.from_records(act)
a.to_csv('log-extracted/action_extractions_chatgpt_4o.csv', index=None)
show(a)

