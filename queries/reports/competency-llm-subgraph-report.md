# Competency LLM Evaluation (Subgraph Context)

- Generated: 2025-12-04 09:05:52 UTC
- Subgraph depth: 1

## Macro Metrics

- Precision: 0.7488
- Recall: 0.5975
- F1: 0.6390

## Micro Metrics

- Precision: 0.6042
- Recall: 0.2628
- F1: 0.3663

## Per-Question Results

| Question | Precision | Recall | F1 | Gold Rows | Pred Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Which causes are documented for the problem "Low Oil Pressure"? | 1.00 | 1.00 | 1.00 | 6 | 6 |
| Which solutions address the problem "Low Oil Pressure"? | 0.83 | 0.83 | 0.83 | 6 | 6 |
| Which functions are assigned to the Cranking Section and the Accessory Housing Assembly? | 1.00 | 1.00 | 1.00 | 3 | 3 |
| Which functions does "Housing and supporting combustion" depend on? | 1.00 | 1.00 | 1.00 | 17 | 17 |
| Which assemblies include part number "STD-475"? | 1.00 | 0.50 | 0.67 | 10 | 5 |
| Which components are responsible for driving power to accessories? | 1.00 | 1.00 | 1.00 | 1 | 1 |
| Which parts belong to the Accessory Housing Assembly? | 1.00 | 0.75 | 0.86 | 8 | 6 |
| Which system contains the Carburetor Assembly? | 1.00 | 1.00 | 1.00 | 1 | 1 |
| Which documented problems share the same cause? | 0.00 | 0.00 | 0.00 | 6 | 0 |
| Which functions depend on "Generating power", and which functions does "Generating power" depend on? | 0.00 | 0.00 | 0.00 | 91 | 16 |
| Which functions ultimately depend on the functions of the Power Section? | 0.96 | 0.31 | 0.46 | 72 | 23 |
| Which systems are most critical based on how many functions depend on them? | 0.17 | 0.18 | 0.17 | 38 | 42 |
| How many components implement each function? | 0.78 | 0.19 | 0.31 | 72 | 18 |

## Subgraph Retrieval Monitoring

| Question | Retrieval Query | Triples | Term Types | Top Predicates |
| --- | --- | ---: | --- | --- |
| Which causes are documented for the problem "Low Oil Pressure"? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#LowOilPressure, https://w3id.org/zorro#hasCause])` | 233 | subj uri:233; obj literal:15, uri:218 | http://www.w3.org/1999/02/22-rdf-syntax-ns#type (74); http://www.w3.org/2000/01/rdf-schema#subClassOf (57); https://w3id.org/zorro#dependsOn (28); +8 more |
| Which solutions address the problem "Low Oil Pressure"? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#LowOilPressure, https://w3id.org/zorro#hasCause, https://w3id.org/zorro#solves])` | 233 | subj uri:233; obj literal:15, uri:218 | http://www.w3.org/1999/02/22-rdf-syntax-ns#type (74); http://www.w3.org/2000/01/rdf-schema#subClassOf (57); https://w3id.org/zorro#dependsOn (28); +8 more |
| Which functions are assigned to the Cranking Section and the Accessory Housing Assembly? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#AccessoryHousingAssembly, https://w3id.org/zorro#CrankingSection, https://w3id.org/zorro#hasFunction])` | 661 | subj bnode:5, uri:656; obj literal:51, uri:610 | https://w3id.org/zorro#partOf (209); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (155); http://www.w3.org/2000/01/rdf-schema#subClassOf (154); +8 more |
| Which functions does "Housing and supporting combustion" depend on? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#HousingAndSupportingCombustion, https://w3id.org/zorro#dependsOn])` | 683 | subj bnode:1, uri:682; obj literal:20, uri:663 | https://w3id.org/zorro#dependsOn (379); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (87); http://www.w3.org/2000/01/rdf-schema#subClassOf (75); +8 more |
| Which assemblies include part number "STD-475"? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#partNumber, https://w3id.org/zorro#partOf, https://w3id.org/zorro#partnr-STD-475])` | 633 | subj bnode:1, uri:632; obj literal:14, uri:619 | http://www.w3.org/1999/02/22-rdf-syntax-ns#type (265); https://w3id.org/zorro#partOf (255); http://www.w3.org/2000/01/rdf-schema#subClassOf (34); +5 more |
| Which components are responsible for driving power to accessories? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#DrivingPowerToAccessories, https://w3id.org/zorro#hasFunction])` | 780 | subj uri:780; obj literal:20, uri:760 | https://w3id.org/zorro#dependsOn (403); http://www.w3.org/2000/01/rdf-schema#subClassOf (110); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (109); +8 more |
| Which parts belong to the Accessory Housing Assembly? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#AccessoryHousingAssembly, https://w3id.org/zorro#partOf])` | 349 | subj bnode:3, uri:346; obj literal:28, uri:321 | https://w3id.org/zorro#partOf (156); http://www.w3.org/2000/01/rdf-schema#subClassOf (72); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (46); +8 more |
| Which system contains the Carburetor Assembly? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#CarburetorAssembly, https://w3id.org/zorro#System, https://w3id.org/zorro#partOf, https://w3id.org/zorro#type])` | 623 | subj bnode:1, uri:622; obj literal:30, uri:593 | https://w3id.org/zorro#partOf (327); http://www.w3.org/2000/01/rdf-schema#subClassOf (76); https://w3id.org/zorro#dependsOn (69); +8 more |
| Which documented problems share the same cause? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#hasCause])` | 120 | subj uri:120; obj uri:120 | http://www.w3.org/1999/02/22-rdf-syntax-ns#type (61); http://www.w3.org/2000/01/rdf-schema#subClassOf (51); http://www.w3.org/2000/01/rdf-schema#range (4); +1 more |
| Which functions depend on "Generating power", and which functions does "Generating power" depend on? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#Function, https://w3id.org/zorro#GeneratingPower, https://w3id.org/zorro#dependsOn, https://w3id.org/zorro#label, https://w3id.org/zorro#type])` | 1415 | subj uri:1415; obj literal:97, uri:1318 | https://w3id.org/zorro#dependsOn (762); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (101); http://www.w3.org/2000/01/rdf-schema#label (97); +8 more |
| Which functions ultimately depend on the functions of the Power Section? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#Function, https://w3id.org/zorro#PowerSection, https://w3id.org/zorro#dependsOn, https://w3id.org/zorro#hasFunction, https://w3id.org/zorro#label, https://w3id.org/zorro#type])` | 2364 | subj bnode:57, uri:2307; obj literal:336, uri:2028 | https://w3id.org/zorro#dependsOn (804); https://w3id.org/zorro#partOf (432); http://www.w3.org/2000/01/rdf-schema#label (262); +8 more |
| Which systems are most critical based on how many functions depend on them? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#Function, https://w3id.org/zorro#System, https://w3id.org/zorro#dependsOn, https://w3id.org/zorro#hasFunction, https://w3id.org/zorro#label, https://w3id.org/zorro#type])` | 1755 | subj uri:1755; obj literal:104, uri:1651 | https://w3id.org/zorro#dependsOn (796); https://w3id.org/zorro#partOf (308); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (125); +8 more |
| How many components implement each function? | `expand_subgraph(depth=1, seeds=[https://w3id.org/zorro#Component, https://w3id.org/zorro#hasFunction, https://w3id.org/zorro#label, https://w3id.org/zorro#subClassOf])` | 1192 | subj bnode:35, uri:1157; obj literal:32, uri:1160 | https://w3id.org/zorro#partOf (566); http://www.w3.org/2000/01/rdf-schema#subClassOf (200); http://www.w3.org/1999/02/22-rdf-syntax-ns#type (191); +7 more |
