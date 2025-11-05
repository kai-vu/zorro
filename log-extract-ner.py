#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[34]:


df = pd.read_csv('/content/Last Annotated problems Aircraft Data - Blad1 (2).csv')


# In[35]:


training_df = df[(df['LOCATION'].notnull())|(df['PART'].notnull())|(df['TGGEDPROBLEM'].notnull())]
train, test = train_test_split(training_df, test_size=0.2, random_state=42)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


# In[4]:


len(train)


# In[36]:


import csv
import random
import spacy

def process_csv(input_file, skip_rows=0):
    spacy_training_data = []
    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            if index < skip_rows:
                continue  # Skip the first `skip_rows` rows

            text = row['PROBLEM']
            entities = []
            for entity_type in ['LOCATION', 'PART', 'TGGEDPROBLEM']:
                entity = row[entity_type]
                if entity and entity != 'None':  # Check if entity is not empty or 'None'
                    start = text.find(entity)
                    end = start + len(entity)
                    entities.append((start, end, entity_type))
            if entities:  # Only add to training data if entities were found
                spacy_training_data.append((text, {"entities": entities}))
    return spacy_training_data

nlp = spacy.blank("en")

ner = nlp.add_pipe("ner")
ner.add_label("LOCATION")
ner.add_label("PART")
ner.add_label("TGGEDPROBLEM")

input_file = '/content/train.csv'

spacy_training_data = process_csv(input_file)
print(len(spacy_training_data), spacy_training_data[0])


# In[37]:


from spacy.training import Example
from spacy.util import minibatch

examples = []
for text, annot in spacy_training_data:
    doc = nlp.make_doc(text)
    entities = annot.get("entities")
    spans = []
    for start, end, label in entities:
        # if label is None or label == 'None':
        #     continue  # Skip entities with None label

        try:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Failed to create span for '{text}' with label '{label}', skipping this annotation.")
            else:
                spans.append(span)
        except ValueError as e:
            print(f"Error processing '{text}' with label '{label}': {e}")

    if spans:
        try:
            examples.append(Example.from_dict(doc, {"entities": spans}))
        except ValueError as e:
            print(f"Ignoring annotation for '{text}' due to error: {e}")



# In[38]:


nlp.begin_training()
for epoch in range(32):
    random.shuffle(examples)
    for batch in minibatch(examples, size=8):
        nlp.update(batch)

# Save the trained model to a directory
output_dir = "ner_model"
nlp.to_disk(output_dir)
print("Trained NER model saved to:", output_dir)


# In[39]:


# Load the trained model from the directory
loaded_nlp = spacy.load(output_dir)

spacy_test_data = process_csv('/content/test.csv')
# print(len(spacy_test_data))

test_examples = []
for text, annot in spacy_test_data:
    doc = nlp.make_doc(text)
    entities = annot.get("entities")
    spans = []
    for start, end, label in entities:
        # if label is None or label == 'None':
        #     continue  # Skip entities with None label

        try:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Failed to create span for '{text}' with label '{label}', skipping this annotation.")
            else:
                spans.append(span)
        except ValueError as e:
            print(f"Error processing '{text}' with label '{label}': {e}")

    if spans:
        try:
            test_examples.append(Example.from_dict(doc, {"entities": spans}))
        except ValueError as e:
            print(f"Ignoring annotation for '{text}' due to error: {e}")

print(len(test_examples))



# In[40]:


# Evaluate the performance of the model
scores = loaded_nlp.evaluate(test_examples)
print("Evaluation scores:", scores)


# In[41]:


print(df.columns)


# In[43]:


# Filter the dataframe for rows where LOCATION, PART, and TAGGEDPROBLEM are null
raw_df = df[(df['LOCATION'].isnull())&(df['PART'].isnull())&(df['TGGEDPROBLEM'].isnull())]

# Create new columns for LOCATION, PART, and TAGGEDPROBLEM based on the predicted entities
raw_df['Location'] = raw_df['PROBLEM'].apply(lambda prb: ', '.join([ent.text for ent in loaded_nlp(prb).ents if ent.label_ == 'LOCATION']))
raw_df['Part'] = raw_df['PROBLEM'].apply(lambda prb: ', '.join([ent.text for ent in loaded_nlp(prb).ents if ent.label_ == 'PART']))
raw_df['Tagged_problem'] = raw_df['PROBLEM'].apply(lambda prb: ', '.join([ent.text for ent in loaded_nlp(prb).ents if ent.label_ == 'TGGEDPROBLEM']))

# Now, you have separate columns for each label
raw_df[['Location', 'Part', 'Tagged_problem']]


# In[44]:


import pandas as pd

# Assuming `df` is the original dataframe
# And `raw_df` contains the predictions

# Update the LOCATION column with predicted labels where original labels are missing
df['LOCATION'] = df['LOCATION'].combine_first(raw_df['Location'])

# Update the PART column with predicted labels where original labels are missing
df['PART'] = df['PART'].combine_first(raw_df['Part'])

# Update the TGGEDPROBLEM column with predicted labels where original labels are missing
df['TGGEDPROBLEM'] = df['TGGEDPROBLEM'].combine_first(raw_df['Tagged_problem'])


# In[45]:


df = df.drop(columns=['PROBLEM', 'ACTION'])

# Save the updated dataframe to a CSV file
df.to_csv('Problem_extraction_NER.csv', index=False)

print("Final labels saved to 'Problem_extraction_NER.csv'")


# In[ ]:




