#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

synonyms = [
    ["cyl", "cylinder"],
    ["baffle", "baffling"],
    ["intake", "induction"],
    ["line", "pipe", "tube"],
]
synonyms = {s: set(ss) for ss in synonyms for s in ss}

catalog = pd.read_csv("pdf-extracted/parts-catalog.csv").set_index("Part Number")
catalog["Type"] = catalog["Type"].str.lower()

vec = TfidfVectorizer(stop_words="english")


def make_text(cat_slice):
    cols = ["Section", "Figure", "Type", "Specifics"]
    return cat_slice[cols].fillna("").apply(" ".join, axis=1)


vec.fit(make_text(catalog))


# In[2]:


from sklearn.metrics.pairwise import cosine_similarity as cos


def expand_synonyms(s):
    s = set(s.split())
    for w in list(s):
        s |= synonyms.get(w, set())
    return " ".join(s)


def root_type_candidates(part_root):
    # fitler parts based on type
    query = synonyms.get(part_root, set([part_root]))
    return catalog[catalog["Type"].apply(lambda x: bool(query & set(x.split())))].index


def match_part(root, rest, score_dropoff_threshold=0.5):
    """score_dropoff_threshold: lower = fewer results"""
    root, rest = root.lower(), rest.lower()

    # the candidates are the parts whose type is the same as the root
    candidates = catalog.loc[root_type_candidates(root)]
    if len(candidates):

        # try to rank the candidates using the TFIDF cosine similarity
        cand_vec = vec.transform(make_text(candidates))
        query_vec = vec.transform([expand_synonyms(rest)])
        score = pd.Series(cos(cand_vec, query_vec).flatten(), index=candidates.index)

        if not score.any():
            # if the rest of the name does not match, check ALL of the text in the figure
            def all_figure_text(x):
                return " ".join(make_text(catalog[catalog["Figure"] == x["Figure"]]))

            fig_text = candidates.apply(all_figure_text, axis=1)
            fig_vec = vec.transform(fig_text)
            score = pd.Series(cos(fig_vec, query_vec).flatten(), index=candidates.index)

        score.sort_values(ascending=False, inplace=True)
        # keep results above score dropoff
        cutoff = ((score.diff(-1) / score) > score_dropoff_threshold).argmax()
        return score.iloc[: cutoff + 1].rename("score")


match_part("GASKET", "ROCKER COVER")
match_part("GASKET", "INDUCTION TUBE")
match_part("SCREW", "ROCKER BOX COVER")
# match_part('CYL','')
# match_part('CRANKSHAFT','')
# match_part('LINE','OIL RETURN')


# In[8]:


from sklearn.metrics.pairwise import cosine_similarity as cos

figures = catalog["Figure"].unique()
sections = catalog["Section"].unique()

fig_vec = vec.transform(figures)
sec_vec = vec.transform(sections)


def match_figure_or_section(root, rest, score_dropoff_threshold=0.5):
    """score_dropoff_threshold: lower = fewer results"""
    root, rest = root.lower(), rest.lower()
    query_vec = vec.transform([expand_synonyms(root + " " + rest)])
    for names, vecs in [(figures, fig_vec), (sections, sec_vec)]:
        score = pd.Series(cos(vecs, query_vec).flatten(), index=names)
        if score.any():
            score.sort_values(ascending=False, inplace=True)
            # keep results above score dropoff
            cutoff = ((score.diff(-1) / score) > score_dropoff_threshold).argmax()
            return score.iloc[: cutoff + 1].rename("score")


match_figure_or_section("CYL", "")


# In[12]:


import re, tqdm, json

fname = "log-extracted/problem_extractions_regex.csv"
parts = pd.read_csv(fname)["part"].dropna()
tree = pd.Series(parts).str.rsplit(" ", n=1, expand=True)
tree = tree.apply(lambda x: pd.Series([x[1] or x[0], x[0] if x[1] else ""]), axis=1)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
tree[0] = tree[0].str.lower().apply(lemmatizer.lemmatize, pos="n").str.upper()

with open("part-links/part-links-regex.tsv", "w") as fw:
    for root, rest in tqdm.tqdm(tree.value_counts().index):
        name = ((rest + " " + root) if rest else root).lower()
        score = match_part(root, rest)
        if score is not None:
            score = (score / score.sum()).fillna(1)
            score_dict = {"partnr-" + k: round(v, 5) for k, v in score.items()}
            print(name, json.dumps(score_dict), sep="\t", file=fw)
        else:
            score = match_figure_or_section(root, rest)
            if score is not None:
                score = (score / score.sum()).fillna(1)
                score_dict = {
                    re.sub("[^a-zA-Z]", "", k): round(v, 5) for k, v in score.items()
                }
                print(name, json.dumps(score_dict), sep="\t", file=fw)
            else:
                print(name, {}, sep="\t", file=fw)
