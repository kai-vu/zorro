# Zorro Maintenance KG Proof-of-Concept

This project aims to explore approaches for constructing, querying and visualizing a Knowledge Graph (KG) for industrial maintenance applications.

It constitutes a proof-of-concept implementation in the domain of aircraft engine maintenance, using data from [MaintNet](https://people.rit.edu/fa3019/MaintNet/data_aviation.html) ([Akhbardeh et al.](#maintnet)) maintenance records (logbooks / log sheets). Observing that this data primarily concerned Lycoming engines from the [University of North Dakota Aviation Program](https://aero.und.edu/aviation/we-offer/airplanes.html), we constructed a KG with a logical, physical, and functional view of the [Lycoming O-320](https://en.wikipedia.org/wiki/Lycoming_O-320) engine, as well as associated troubleshooting information.

This way, we can integrate extractions from the historical records with maintenance knowledge to gain insights into failure frequencies, causes, and patterns.

The KG structure can be seen as 3 layers: the *schema*, *domain knowledge*, and *historical records*, which are described below.

## Schema
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
We are interested in mentioned parts, their location (engine and cylinder), and the failure type.
To do this we use 3 approaches: regular expressions, NER, and GPT4.

### Part links
We link extracted part names to parts from the part catalog.
To do this, we use a Filtered Contextual Bag-of-Words matching approach.
For every unique extracted part name, we try to find the most likely matching candidates from the part catalog.
First, we try to construct a set of candidates by matching the last word of the mentioned part name (the head of the noun phrase) ...

## References
### MaintNet
Akhbardeh, F., Desell, T., & Zampieri, M. (2020, December). MaintNet: A Collaborative Open-Source Library for Predictive Maintenance Language Resources. In Proceedings of the 28th International Conference on Computational Linguistics: System Demonstrations (pp. 7-11).