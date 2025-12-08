# AKGAM

We introduce AKGAM, a publicly available resource designed to address these gaps. AKGAM includes: (i) an LLM-assisted, manually verified annotated corpus of
aircraft maintenance logbooks; (ii) an ontology-governed, provenance-rich fault-diagnosis KG built from publicly available datasets, developed and validated with CPS domain experts; and (iii) a set of competency questions (CQs) that formalize core diagnostic and troubleshooting tasks. AKGAM provides triple-level provenance, a reproducible population pipeline, and a lightweight CQ execution interface. By offering the first CPS KG with full schema governance and end-to-end provenance, AKGAM enables systematic development and benchmarking of KG-based diagnostic methods under realistic CPS conditions.

The KG structure can be seen as 3 layers: the *schema*, *domain knowledge*, and *historical records*, which are described below.

## Build & Validation

- Run the end-to-end workflow with `python -m src.build_all`. The makeprov CLI orchestrates the prompt extractions, regex extractions, part linking (SSSOM TSV + TriG) and RDF graph synthesis, skipping steps that are already up to date.
- Individual stages can be executed directly by calling the decorated rules, for example `python -m log_extract_regex`, `python -m log_extract_gpt` (uses cached OpenAI completions by default), `python -m log_extract_ner` (trains + runs spaCy), or `python -m make_rdf`.
- Validate outputs with `python -m pytest`, which parses the generated TriG datasets and the SSSOM TSV to ensure schema compliance.

## Query Exploration

- Run `python -m src.sparql_bar_chart` to execute the default top-problem query (`queries/logbook/01-top10-problem-types.rq`) against the GPT extraction graph and render a horizontal bar chart (`query_bar_chart.png`).
- Tweak the visualization with options such as `--top-n`, `--orientation`, `--wrap-width`, or `--style` (see `python -m src.sparql_bar_chart --help`).
- Produce a markdown regression report for all competency questions with `python -m src.competency_report`. Use `--report-path` to control the output location (defaults to `queries/reports/competency-report.md`). Queries are grouped under `queries/documentation/` (documentation-derived knowledge) and `queries/logbook/` (logbook-derived knowledge).
- Benchmark LLM answers to documentation-focused competency questions with `python -m src.competency_llm_eval`. By default it extracts depth-1 subgraphs per question; add `--depth N` to widen the neighbourhood, `--full-graph` to fall back to the legacy whole-graph mode, `--model NAME` to pick a different API deployment, `--lenience FRACTION` to permit proportional per-row mismatches during scoring, `--cache-only` to avoid new calls, or `--profile-subgraphs` to dump only the retrieved subgraphs (Turtle under a depth/model/lenience-specific directory) plus JSON telemetry. Outputs are written to `queries/reports/competency_llm_depth-<depth|full>_model-<model>_lenience-<fraction>_{cache,results,report}.json/md`, which capture the retrieval request, triple breakdown, lenience setting, and evaluation metrics for reproducibility.

## Schema
The ontology specification draft is available at https://w3id.org/ZorroOntology.
The paper that explain the construction of the schema is available at https://ceur-ws.org/Vol-3830/paper1sim.pdf
(see paper)

## Domain Knowledge
We extracted two tables from documents about the Lycoming O-320 engine:

- **Parts Catalog** (from PDF)<br>
Provides full list of parts, in a physical and logical hierarchy. The catalog is structured into sections which correspond to physical (sub)systems of the engine, which contain figures that describe assemblies. Each part description contains the logical type of the part, and has a unique part number.

- **Troubleshooting** (from operator manual PDF)<br>
A table describing frequent observable troubles, possible causes, and remedies.

Then, we enriched this data by treating *GPT4 as a Proxy Expert*, asking it for knowledge that would further integrate this information. The prompt first gives some background information on the engine (from the operator manual), and then asks for structured output about:

- the functions of assemblies and systems in the parts catalog,
- the functional dependencies of systems and assemblies on one another,
- links between problems and components,
- and a classification scheme for part types.

## Historical Records
We extract information from [**MaintNet**](https://people.rit.edu/fa3019/MaintNet/data_aviation.html) ([Akhbardeh et al.](#maintnet)) records of free text fields describing problems and actions.

### Part links
We link extracted part names to parts from the part catalog.

One of the methods is called Filtered Contextual Bag-of-Words matching.
For every unique extracted part name, we try to find the most likely matching candidates from the part catalog.
First, we try to construct a set of candidates by (exactly) matching the (lemmatized) *last word* (such as "gasket") of the mentioned part name (the head of the noun phrase) to a *type* from the parts catalog (this is the filter).

- If this fails, we return the similarities of the part name with the names of systems and assemblies.
- Else, the set of candidates consists of all parts of the matched type.
We then calculate the similarities of the extracted part name with the candidate descriptions (including the names of their assemblies and systems).
    - If this gives any non-zero similarities, we return those.
    - Else, we collect the descriptions of all neighboring parts in the same assembly for each candidate part. Then we return the similarities between the extracted part name and these texts.

The text similarity function we use is the cosine similarity of TFIDF bag-of-words vectors.

## Annotated Corpus
We construct an annotated corpus from real aircraft engine maintenance logbooks. Each record in this dataset consists of an identifier, a free-text problem description, and the corresponding maintenance action.  To support KG construction, we extract five diagnostic entities from each entry: problem type, faulty component, location, action type, and action part. These labels capture the essential information required for downstream semantic modelling. To establish ground truth, 500 records were manually annotated. These examples served as ten-shot in-context prompt demonstrations for GPT-4.1-mini, which annotated the remaining 5,669 records. All the annotation were then manually reviewed and corrected, yielding a fully validated corpus suitable as  a ground truth. 


## Citation
If you use the resources presented in this repository, please cite:

```bibtex
```
## Contact
Should you have any questions, please contact Ameneh Naghdipour at a.naghdipour@vu.nl.




### References
- NaghdiPour, A., Kruit, B., Chen,J., and Schlobach, S. (2024). Knowledge Representation and Engineering for Smart Diagnosis of Cyber Physical Systems. SOFLIM2KG-SemIIM@ISWC.
- NaghdiPour, A., Kruit, B., Chen,J., and Schlobach, S. (2024). Intelligent Fault Diagnosis of Cyber Physical Systems using Knowledge Graphs. International Conference Knowledge Engineering and Knowledge Management.
- Akhbardeh, F., Desell, T., & Zampieri, M. (2020, December). MaintNet: A Collaborative Open-Source Library for Predictive Maintenance Language Resources. In Proceedings of the 28th International Conference on Computational Linguistics: System Demonstrations (pp. 7-11).
