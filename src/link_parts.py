"""Link extracted maintenance parts to catalog entries and emit SSSOM artefacts.

This module exposes `makeprov`-compatible rules that (1) score part mentions
against the catalog using TF-IDF similarity, (2) emit SSSOM-compliant TSV files
with rich provenance metadata, and (3) serialize the mappings as RDF named
graphs.  The implementation keeps the existing heuristics from the legacy
notebook script, but restructures them into testable functions with proper
logging, dataclasses, and provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from makeprov import InPath, OutPath, rule
from rdflib import BNode, ConjunctiveGraph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, RDF, RDFS, SKOS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sssom.util import MappingSetDataFrame
from sssom.writers import to_rdf_graph, write_tsv

LOGGER = logging.getLogger(__name__)

BASE_IRI = "https://w3id.org/zorro#"
BASE = Namespace(BASE_IRI)
GRAPH_NS = Namespace("https://w3id.org/zorro/graph/")
SEMAPV = Namespace("http://purl.org/semapv/")


def _slugify(value: str) -> str:
    """Return a filesystem and IRI friendly slug."""
    clean = "".join(char if char.isalnum() else " " for char in value)
    return "_".join(chunk for chunk in clean.lower().split() if chunk)


def _compute_sha256(path: Path) -> str:
    """Return the SHA256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lemmatize_head(token: str) -> str:
    """Light-weight lemmatizer for noun heads (avoids nltk downloads)."""
    token = token.lower()
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent types and helper columns on the catalog."""
    catalog = catalog.copy()
    catalog.columns = [c.strip().replace(" ", "_").lower() for c in catalog.columns]
    catalog["type"] = catalog["type"].astype(str).str.lower()
    catalog["specifics"] = catalog["specifics"].fillna("").astype(str)
    catalog["label"] = (
        catalog["specifics"].str.strip() + " " + catalog["type"].str.upper()
    ).str.strip()
    catalog["part_number"] = catalog["part_number"].astype(str)
    catalog.set_index("part_number", inplace=True, drop=False)
    return catalog


def _tfidf_vectorizer(catalog: pd.DataFrame) -> TfidfVectorizer:
    """Build and fit a TF-IDF vectorizer on catalog descriptive text."""
    vec = TfidfVectorizer(stop_words="english")
    descriptive_cols = ["section", "figure", "type", "specifics"]
    vec.fit(catalog[descriptive_cols].fillna("").apply(" ".join, axis=1))
    return vec


def _expand_synonyms(tokens: Iterable[str], synonyms: Mapping[str, set[str]]) -> set[str]:
    """Expand tokens with synonym sets."""
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        expanded.update(synonyms.get(token, set()))
    return expanded


def _score_candidates(
    mention: str,
    head: str,
    catalog: pd.DataFrame,
    vec: TfidfVectorizer,
    synonyms: Mapping[str, set[str]],
    threshold: float,
) -> Mapping[str, float]:
    """Return candidate part numbers with normalized similarity scores."""
    mention_tokens = mention.lower().split()
    expanded = _expand_synonyms(mention_tokens, synonyms)
    query_text = " ".join(sorted(expanded))
    query_vec = vec.transform([query_text])

    mask = catalog["type"].apply(lambda x: _lemmatize_head(x) == head)
    candidates = catalog.loc[mask]

    if candidates.empty:
        return {}

    cand_vec = vec.transform(
        candidates[["section", "figure", "type", "specifics"]].fillna("").apply(
            " ".join, axis=1
        )
    )

    scores = cosine_similarity(cand_vec, query_vec).reshape(-1)
    series = pd.Series(scores, index=candidates["part_number"])
    series = series[series > 0]
    if series.empty:
        return {}

    series = series.sort_values(ascending=False)
    diffs = series.diff(-1).abs()
    if not diffs.empty:
        diff_ratio = diffs.divide(series, fill_value=0).to_numpy()
        drop_positions = (diff_ratio > threshold).nonzero()[0]
        if drop_positions.size:
            cutoff_position = int(drop_positions[0] + 1)
            series = series.iloc[:cutoff_position]

    normalized = series / series.sum()
    return normalized.to_dict()


@dataclass(slots=True)
class LinkingConfig:
    """Runtime configuration for part linking."""

    synonyms: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "cyl": ("cylinder",),
            "baffle": ("baffling",),
            "intake": ("induction",),
            "line": ("pipe", "tube"),
        }
    )
    score_dropoff_threshold: float = 0.5
    predicate_id: str = "skos:closeMatch"
    mapping_tool: str = "filtered-tfidf"
    mapping_tool_version: str = "0.1"
    mapping_justification: str = "semapv:LexicalMatching"
    mapping_association: str = "semapv:AutomatedMapping"

    def synonym_lookup(self) -> dict[str, set[str]]:
        """Return synonyms as lowercase lookup table."""
        lookup: dict[str, set[str]] = {}
        for key, values in self.synonyms.items():
            head = _lemmatize_head(key.lower())
            expanded = {head, key.lower()}
            expanded.update(_lemmatize_head(v.lower()) for v in values)
            lookup[head] = expanded
        return lookup


def _part_mentions(extraction: pd.DataFrame) -> pd.Series:
    """Return unique part mentions from the extraction frame."""
    parts = extraction["part"].dropna().astype(str).str.strip()
    parts = parts[parts.ne("")]
    return parts.drop_duplicates().reset_index(drop=True)


def _head_modifier_split(part_name: str) -> tuple[str, str]:
    """Split a part mention into head noun and modifiers."""
    tokens = part_name.split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return _lemmatize_head(tokens[0]), ""
    head = _lemmatize_head(tokens[-1])
    modifiers = " ".join(tokens[:-1])
    return head, modifiers


def _build_prefix_map() -> dict[str, str]:
    """Return prefix map for SSSOM artefacts."""
    return {
        "zorro": BASE_IRI,
        "skos": str(SKOS),
        "semapv": str(SEMAPV),
        "prov": str(PROV),
        "dcterms": str(DCTERMS),
        "owl": str(OWL),
    }


def _mapping_id(subject_slug: str, object_curie: str) -> str:
    """Return deterministic mapping identifier."""
    return f"zorro:mapping/{subject_slug}--{_slugify(object_curie)}"


def _subject_curie(subject_slug: str) -> str:
    """Return subject CURIE."""
    return f"zorro:part-mention/{subject_slug}"


def _object_curie(part_number: str) -> str:
    """Return object CURIE based on catalog part number."""
    return f"zorro:partnr-{part_number}"


def _mapping_rows(
    mentions: Iterable[str],
    catalog: pd.DataFrame,
    vec: TfidfVectorizer,
    config: LinkingConfig,
    source_path: Path,
) -> list[dict[str, str | float]]:
    """Score mentions and yield SSSOM mapping rows.

    Args:
        mentions: Unique part mention strings.
        catalog: Normalized catalog dataframe.
        vec: TF-IDF vectorizer fitted on the catalog.
        config: Linking configuration parameters.
        source_path: Path to the extraction file (for provenance).

    Returns:
        List of dictionaries representing SSSOM rows.
    """
    rows: list[dict[str, str | float]] = []
    synonym_lookup = config.synonym_lookup()
    checksum = _compute_sha256(source_path)
    mapping_date = date.today().isoformat()
    tool = f"{config.mapping_tool} v{config.mapping_tool_version}"

    for mention in mentions:
        head, modifiers = _head_modifier_split(mention.lower())
        if not head:
            LOGGER.debug("Skipping empty mention %r", mention)
            continue
        scores = _score_candidates(
            mention,
            head,
            catalog,
            vec,
            synonym_lookup,
            config.score_dropoff_threshold,
        )
        if not scores:
            LOGGER.info("No candidate for mention %s", mention)
            continue

        subject_slug = _slugify(mention)
        subject_curie = _subject_curie(subject_slug)
        subject_label = mention

        for part_number, confidence in scores.items():
            catalog_row = catalog.loc[part_number]
            if isinstance(catalog_row, pd.DataFrame):
                catalog_entry = catalog_row.iloc[0]
            else:
                catalog_entry = catalog_row
            object_curie = _object_curie(part_number)
            rows.append(
                {
                    "mapping_id": _mapping_id(subject_slug, part_number),
                    "subject_id": subject_curie,
                    "subject_label": subject_label,
                    "object_id": object_curie,
                    "object_label": catalog_entry["label"],
                    "predicate_id": config.predicate_id,
                    "mapping_date": mapping_date,
                    "mapping_justification": config.mapping_justification,
                    "confidence": float(np.round(confidence, 5)),
                    "mapping_tool": tool,
                    "mapping_association": config.mapping_association,
                    "mapping_source": str(source_path),
                    "mapping_source_checksum": checksum,
                    "mapping_source_column": "part",
                    "mapping_notes": modifiers or "",
                }
            )
    return rows


def _as_msdf(rows: Sequence[dict[str, str | float]]) -> MappingSetDataFrame:
    """Convert raw rows into a MappingSetDataFrame with prefixes.

    Args:
        rows: Iterable of mapping row dictionaries.

    Returns:
        MappingSetDataFrame ready for serialization.
    """
    df = pd.DataFrame(rows)
    metadata = {
        "mapping_set_id": f"{BASE_IRI}mapping-set/part-links-regex",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "creator_id": "zorro:link_parts.py",
        "creator_label": "Zorro Part Linking Script",
    }
    msdf = MappingSetDataFrame(df=df, metadata=metadata)
    msdf.prefix_map.update(_build_prefix_map())
    return msdf


def _write_rdf_graph(
    msdf: MappingSetDataFrame,
    trig_out: OutPath,
) -> ConjunctiveGraph:
    """Serialize MappingSetDataFrame as TriG with named graph.

    Args:
        msdf: Mapping set data frame to serialize.
        trig_out: Output TriG path.

    Returns:
        ConjunctiveGraph containing the named graph.
    """
    dataset = ConjunctiveGraph()
    graph_iri = GRAPH_NS["part-links-sssom"]
    named_graph = dataset.get_context(graph_iri)

    sssom_graph = to_rdf_graph(msdf)
    for triple in sssom_graph:
        named_graph.add(triple)

    # Attach provenance metadata per mapping.
    for _, row in msdf.df.iterrows():
        mapping_node = URIRef(row["mapping_id"])
        activity_node = BNode()
        named_graph.add((mapping_node, RDF.type, OWL.Axiom))
        named_graph.add((mapping_node, PROV.wasGeneratedBy, activity_node))
        named_graph.add((mapping_node, DCTERMS.date, Literal(row["mapping_date"])))
        named_graph.add(
            (mapping_node, DCTERMS.source, Literal(row["mapping_source"]))
        )
        named_graph.add(
            (
                mapping_node,
                DCTERMS.description,
                Literal(row.get("mapping_notes", "")),
            )
        )
        named_graph.add(
            (
                activity_node,
                PROV.startedAtTime,
                Literal(datetime.now(timezone.utc)),
            )
        )
        named_graph.add(
            (
                activity_node,
                PROV.wasAssociatedWith,
                URIRef(f"{BASE_IRI}agent/part-linker"),
            )
        )
        named_graph.add(
            (
                activity_node,
                PROV.used,
                Literal(row.get("mapping_tool", "")),
            )
        )
        named_graph.add(
            (
                activity_node,
                DCTERMS.identifier,
                Literal(row["mapping_source_checksum"]),
            )
        )

    dataset.serialize(destination=str(trig_out), format="trig")
    return dataset


@rule(
    name="part_links_regex",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def build_part_links_regex(
    catalog_csv: InPath = InPath("pdf-extracted/parts-catalog.csv"),
    extraction_csv: InPath = InPath("log-extracted/problem_extractions_regex.csv"),
    sssom_tsv: OutPath = OutPath("part-links/part-links-regex.sssom.tsv"),
    sssom_trig: OutPath = OutPath("generated-rdf/part-links-regex.sssom.trig"),
    *,  # force keyword-only for config
    config: LinkingConfig = LinkingConfig(),
):
    """Create SSSOM alignment for regex-derived part mentions.

    Args:
        catalog_csv: Path to the parts catalog CSV file.
        extraction_csv: Path to the regex extraction CSV with `part` column.
        sssom_tsv: Output path for the SSSOM TSV.
        sssom_trig: Output path for the SSSOM TriG named graph.
        config: Runtime tuning parameters for linking.

    Returns:
        RDF dataset containing the SSSOM named graph.
    """
    catalog_path = Path(catalog_csv)
    extraction_path = Path(extraction_csv)

    LOGGER.info("Loading catalog from %s", catalog_path)
    catalog = _normalize_catalog(pd.read_csv(catalog_path))
    vectorizer = _tfidf_vectorizer(catalog)

    LOGGER.info("Loading part mentions from %s", extraction_path)
    extraction_df = pd.read_csv(extraction_path)
    mentions = _part_mentions(extraction_df)

    LOGGER.info("Scoring %d mentions", len(mentions))
    rows = _mapping_rows(
        mentions=mentions,
        catalog=catalog,
        vec=vectorizer,
        config=config,
        source_path=extraction_path,
    )
    if not rows:
        LOGGER.warning("No mapping rows produced; nothing to write.")
        return None

    msdf = _as_msdf(rows)
    LOGGER.info("Writing SSSOM TSV to %s", sssom_tsv)
    Path(sssom_tsv).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(msdf, Path(sssom_tsv))

    LOGGER.info("Serializing SSSOM RDF to %s", sssom_trig)
    Path(sssom_trig).parent.mkdir(parents=True, exist_ok=True)
    return _write_rdf_graph(msdf, sssom_trig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_part_links_regex()
