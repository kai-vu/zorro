"""Generate RDF named graphs for the Zorro maintenance knowledge graph."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
import pandas as pd
from makeprov import InPath, OutPath, build, rule
from rdflib import ConjunctiveGraph, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DC, DCTERMS, OWL, RDF, RDFS, SKOS

LOGGER = logging.getLogger(__name__)

BASE_IRI = "https://w3id.org/zorro#"
BASE = Namespace(BASE_IRI)
GRAPH_NS = Namespace("https://w3id.org/zorro/graph/")


def iri_slug(value: str) -> str:
    """Return a compact slug suitable for IRIs."""
    clean = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"\s+", "", clean.title())


def _new_dataset(graph_suffix: str) -> tuple[ConjunctiveGraph, Graph]:
    """Create a dataset with a named graph for the suffix."""
    dataset = ConjunctiveGraph()
    graph = dataset.get_context(GRAPH_NS[graph_suffix])
    graph.bind("", BASE)
    graph.bind("dc", DC)
    graph.bind("dcterms", DCTERMS)
    graph.bind("skos", SKOS)
    graph.bind("owl", OWL)
    return dataset, graph


def _write_dataset(dataset: ConjunctiveGraph, output: OutPath) -> ConjunctiveGraph:
    """Serialize dataset to TriG and return it."""
    dataset.serialize(destination=str(output), format="trig")
    return dataset


@dataclass(slots=True)
class TroubleshootingConfig:
    """Configuration for troubleshooting graph generation."""

    base: Namespace = BASE


@rule(
    name="troubleshooting_trig",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def build_troubleshooting_graph(
    troubleshooting_csv: InPath = InPath("pdf-extracted/troubleshooting.csv"),
    trig_out: OutPath = OutPath("generated-rdf/troubleshooting.trig"),
    *,
    config: TroubleshootingConfig = TroubleshootingConfig(),
) -> ConjunctiveGraph:
    """Convert troubleshooting CSV to an RDF named graph.

    Args:
        troubleshooting_csv: Path to CSV with troubleshooting records.
        trig_out: Target TriG file path.
        config: Troubleshooting configuration.

    Returns:
        Serialized RDF dataset containing the troubleshooting graph.
    """
    dataset, graph = _new_dataset("troubleshooting")
    with open(troubleshooting_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            trouble = config.base[iri_slug(row["TROUBLE"])]
            cause = config.base[iri_slug(row["PROBABLE CAUSE"])]
            remedy = config.base[iri_slug(row["REMEDY"])]

            graph.add((trouble, RDF.type, config.base.Problem))
            graph.add((trouble, RDFS.label, Literal(row["TROUBLE"])))
            graph.add((trouble, config.base.hasCause, cause))

            graph.add((cause, RDF.type, config.base.Cause))
            graph.add((cause, RDFS.label, Literal(row["PROBABLE CAUSE"])))

            graph.add((remedy, RDF.type, config.base.Remedy))
            graph.add((remedy, RDFS.label, Literal(row["REMEDY"])))
            graph.add((remedy, config.base.addresses, trouble))

    return _write_dataset(dataset, trig_out)


@dataclass(slots=True)
class PartsConfig:
    """Configuration for parts graph generation."""

    base: Namespace = BASE


def _ensure_superclass_chain(
    graph: Graph, cls: URIRef, lookup: dict[URIRef, URIRef], base: Namespace
) -> None:
    """Ensure superclass hierarchy is closed."""
    parent = lookup.get(cls)
    if parent:
        graph.add((cls, RDFS.subClassOf, parent))
        _ensure_superclass_chain(graph, parent, lookup, base)
    else:
        graph.add((cls, RDFS.subClassOf, base.Part))


@rule(
    name="parts_trig",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def build_parts_graph(
    part_classes_tsv: InPath = InPath("prompt-extracted/part-classes.tsv"),
    catalog_csv: InPath = InPath("pdf-extracted/parts-catalog.csv"),
    trig_out: OutPath = OutPath("generated-rdf/parts.trig"),
    *,
    config: PartsConfig = PartsConfig(),
) -> ConjunctiveGraph:
    """Convert part classes and catalog tables into an RDF named graph.

    Args:
        part_classes_tsv: TSV describing part subclass relationships.
        catalog_csv: Catalog CSV with detailed part records.
        trig_out: Target TriG file path.
        config: Parts configuration.

    Returns:
        Serialized RDF dataset containing the parts graph.
    """
    dataset, graph = _new_dataset("parts")
    lookup: dict[URIRef, URIRef] = {}

    part_classes = pd.read_csv(part_classes_tsv, sep="\t")
    for _, row in part_classes.iterrows():
        part = config.base[iri_slug(row["Part"])]
        superclass = config.base[iri_slug(row["subClassOf"])]
        lookup[part] = superclass
        graph.add((part, RDF.type, config.base.Part))
        graph.add((part, RDFS.label, Literal(row["Part"])))
        graph.add((part, RDFS.subClassOf, superclass))
        graph.add((superclass, RDFS.label, Literal(row["subClassOf"])))

    for _, row in pd.read_csv(catalog_csv).iterrows():
        system = config.base[iri_slug(row["Section"])]
        assembly = config.base[iri_slug(row["Figure"])]
        part_cls = config.base[iri_slug(row["Type"])]
        part_uri = config.base[f'partnr-{row["Part Number"]}']

        graph.add((system, RDF.type, config.base.System))
        graph.add((system, RDFS.label, Literal(row["Section"])))
        graph.add((assembly, RDF.type, config.base.Assembly))
        graph.add((assembly, RDFS.label, Literal(row["Figure"])))

        graph.add((part_uri, RDF.type, config.base.PartInstance))
        graph.add((part_uri, RDFS.label, Literal(row["Specifics"] or row["Type"])))
        graph.add((part_uri, config.base.partNumber, Literal(row["Part Number"])))
        graph.add((part_uri, config.base.partOf, assembly))
        graph.add((part_uri, config.base.partOf, system))
        graph.add((part_uri, RDFS.subClassOf, part_cls))

        if part_cls not in lookup:
            lookup[part_cls] = config.base.Part
        _ensure_superclass_chain(graph, part_cls, lookup, config.base)

    return _write_dataset(dataset, trig_out)


@dataclass(slots=True)
class FunctionsConfig:
    """Configuration for function graph generation."""

    base: Namespace = BASE


def _add_function(graph: Graph, function_label: str, base: Namespace) -> URIRef:
    """Add a function individual and return its URI."""
    func_uri = base[iri_slug(function_label)]
    graph.add((func_uri, RDF.type, base.Function))
    graph.add((func_uri, RDFS.label, Literal(function_label)))
    return func_uri


@rule(
    name="functions_trig",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def build_functions_graph(
    problem_component_function_tsv: InPath = InPath(
        "prompt-extracted/problem-component-function.tsv"
    ),
    functions_tsv: InPath = InPath("prompt-extracted/functions.tsv"),
    subfunction_tsv: InPath = InPath("prompt-extracted/subfunction.tsv"),
    depends_on_tsv: InPath = InPath("prompt-extracted/dependsOn.tsv"),
    trig_out: OutPath = OutPath("generated-rdf/functions.trig"),
    *,
    config: FunctionsConfig = FunctionsConfig(),
) -> ConjunctiveGraph:
    """Combine prompt-generated TSV files into an RDF function graph.

    Args:
        problem_component_function_tsv: TSV linking problems to components and functions.
        functions_tsv: TSV mapping components to functions.
        subfunction_tsv: TSV describing sub-function hierarchy.
        depends_on_tsv: TSV summarising component dependencies.
        trig_out: Target TriG file path.
        config: Functions configuration.

    Returns:
        Serialized RDF dataset containing the functions graph.
    """
    dataset, graph = _new_dataset("functions")

    pc_df = pd.read_csv(problem_component_function_tsv, sep="\t")
    for _, row in pc_df.iterrows():
        function_uri = _add_function(graph, row["Function"], config.base)
        problem_uri = config.base[iri_slug(row["defines"])]
        component_uri = config.base[iri_slug(row["functionOf"])]

        graph.add((problem_uri, RDF.type, config.base.Problem))
        graph.add((problem_uri, RDFS.label, Literal(row["defines"])))

        graph.add((component_uri, RDF.type, config.base.Component))
        graph.add((component_uri, RDFS.label, Literal(row["functionOf"])))
        graph.add((component_uri, config.base.hasFunction, function_uri))
        graph.add((function_uri, config.base.addressesProblem, problem_uri))

    func_df = pd.read_csv(functions_tsv, sep="\t")
    for _, row in func_df.iterrows():
        component_uri = config.base[iri_slug(row["Component"])]
        function_uri = _add_function(graph, row["hasFunction"], config.base)
        graph.add((component_uri, config.base.hasFunction, function_uri))

    sub_df = pd.read_csv(subfunction_tsv, sep="\t")
    for _, row in sub_df.iterrows():
        parent = _add_function(graph, row["subFunctionOf"], config.base)
        child = _add_function(graph, row["Function"], config.base)
        graph.add((child, config.base.subFunctionOf, parent))

    dep_df = pd.read_csv(depends_on_tsv, sep="\t")
    for _, row in dep_df.iterrows():
        a = config.base[iri_slug(row["Component"])]
        b = config.base[iri_slug(row["dependsOn"])]
        graph.add((a, config.base.dependsOn, b))

    return _write_dataset(dataset, trig_out)


@dataclass(slots=True)
class ExtractionConfig:
    """Configuration for log extractions."""

    base: Namespace = BASE


def _add_event(
    graph: Graph,
    base: Namespace,
    row: dict[str, str],
    kind: str,
    log_lookup: dict[str, dict[str, str]],
) -> URIRef:
    """Create an event node for problem or action."""
    log = log_lookup.get(row["id"])
    if not log:
        return base[f"{kind.lower()}-unknown"]
    event_uri = base[f'{kind.lower()}-{log["IDENT"]}']
    graph.add((event_uri, RDF.type, base[kind.title()]))
    graph.add((event_uri, RDFS.label, Literal(f'{kind.title()} {log["IDENT"]}')))
    graph.add((event_uri, DC.description, Literal(log[kind.upper()], lang="en")))

    if row.get(kind.lower()):
        state_uri = base[iri_slug(f"{row[kind.lower()]} {kind}")]
        graph.add((event_uri, RDF.type, state_uri))
        graph.add((state_uri, RDFS.subClassOf, base[kind.title()]))
        graph.add((state_uri, RDFS.label, Literal(row[kind.lower()])))

    if row.get("part"):
        part_uri = base[iri_slug(row["part"])]
        graph.add((event_uri, base.involvesPart, part_uri))
        graph.add((part_uri, RDF.type, base.PartMention))
        graph.add((part_uri, RDFS.label, Literal(row["part"])))

    if row.get("cylinders"):
        graph.add((event_uri, base.affectsCylinder, Literal(row["cylinders"])))
    if row.get("engine"):
        graph.add((event_uri, base.affectsEngine, Literal(row["engine"])))

    return event_uri


def _load_log_index(log_csv: InPath) -> dict[str, dict[str, str]]:
    """Index maintenance logs by identifier."""
    index: dict[str, dict[str, str]] = {}
    with open(log_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            index[row["IDENT"]] = row
    return index


def _load_extraction_rows(path: InPath) -> list[dict[str, str]]:
    """Load extraction CSV rows as dictionaries."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _build_extraction_graph(
    problems_csv: InPath,
    actions_csv: InPath,
    logs_csv: InPath,
    trig_out: OutPath,
    graph_suffix: str,
    config: ExtractionConfig,
) -> ConjunctiveGraph:
    """Shared logic for building extraction graphs."""
    dataset, graph = _new_dataset(graph_suffix)
    log_index = _load_log_index(logs_csv)
    problem_rows = _load_extraction_rows(problems_csv)
    action_rows = _load_extraction_rows(actions_csv)

    for row in problem_rows:
        _add_event(graph, config.base, row, "problem", log_index)
    for row in action_rows:
        action_event = _add_event(graph, config.base, row, "action", log_index)
        problem_uri = config.base[f'problem-{row["id"]}']
        graph.add((action_event, config.base.dealsWith, problem_uri))

    return _write_dataset(dataset, trig_out)


@rule(
    name="extractions_regex_trig",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def build_extractions_regex_graph(
    problems_csv: InPath = InPath("log-extracted/problem_extractions_regex.csv"),
    actions_csv: InPath = InPath("log-extracted/action_extractions_regex.csv"),
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    trig_out: OutPath = OutPath("generated-rdf/extractions_regex.trig"),
    *,
    config: ExtractionConfig = ExtractionConfig(),
) -> ConjunctiveGraph:
    """Build extraction graph for regex-based pipeline outputs."""
    return _build_extraction_graph(
        problems_csv=problems_csv,
        actions_csv=actions_csv,
        logs_csv=logs_csv,
        trig_out=trig_out,
        graph_suffix="extractions-regex",
        config=config,
    )


@rule(
    name="extractions_gpt_trig",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def build_extractions_gpt_graph(
    problems_csv: InPath = InPath("log-extracted/problem_extractions_chatgpt_4o.csv"),
    actions_csv: InPath = InPath("log-extracted/action_extractions_chatgpt_4o.csv"),
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    trig_out: OutPath = OutPath("generated-rdf/extractions_chatgpt_4o.trig"),
    *,
    config: ExtractionConfig = ExtractionConfig(),
) -> ConjunctiveGraph:
    """Build extraction graph for GPT-based pipeline outputs."""
    dataset = _build_extraction_graph(
        problems_csv=problems_csv,
        actions_csv=actions_csv,
        logs_csv=logs_csv,
        trig_out=trig_out,
        graph_suffix="extractions-gpt",
        config=config,
    )
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build("generated-rdf/troubleshooting.trig")
    build("generated-rdf/parts.trig")
    build("generated-rdf/functions.trig")
    build("generated-rdf/extractions_regex.trig")
    build("generated-rdf/extractions_chatgpt_4o.trig")
