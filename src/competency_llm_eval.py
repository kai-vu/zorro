"""Evaluate documentation competency questions with optional subgraph contexts.

This module unifies the previous full-graph and subgraph evaluators. By default it
extracts question-specific neighbourhoods, but a `--full-graph` flag switches to the
legacy mode that serialises the entire dataset. Outputs (cache, results, reports,
profiles) are now tagged with the effective subgraph depth and model name.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI
from rdflib import ConjunctiveGraph, Graph, Namespace, URIRef
from rdflib.namespace import RDFS
from rdflib.term import BNode, Literal


DEFAULT_MODEL = "gpt-5-nano"
API_KEY_PATH = Path("openai-key.txt")
QUERY_DIR = Path("queries")
DOCUMENTATION_QUERY_DIR = QUERY_DIR / "documentation"
GENERATED_RDF_DIR = Path("generated-rdf")
EXCLUDED_PREFIXES = ("extractions_",)
REPORT_DIR = Path("queries/reports")
LEGACY_CACHE_PATHS = [
    REPORT_DIR / "competency_llm_subgraph_cache.json",
    REPORT_DIR / "competency_llm_cache.json",
]
BASE_NS = Namespace("https://w3id.org/zorro#")
DEFAULT_SUBGRAPH_DEPTH = 1
DEFAULT_LENIENCE = 0.0

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputPaths:
    """Filesystem destinations for a particular (model, depth) evaluation run."""

    cache: Path
    summary: Path
    report: Path
    profile_summary: Path
    profile_dir: Path


@dataclass(frozen=True)
class SubgraphMonitoring:
    """Summarise the retrieval request and contents of a depth-limited subgraph."""

    retrieval_query: str
    depth: int | None
    seeds: list[str]
    triple_count: int
    predicate_breakdown: list[dict[str, int]]
    subject_term_types: dict[str, int]
    object_term_types: dict[str, int]

    def asdict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary representation.

        Returns:
            dict[str, object]: Dictionary payload for persistence.
        """
        return asdict(self)


def slugify(value: str) -> str:
    """Generate a filesystem-friendly slug for the given string."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "model"


def build_output_paths(model_name: str, depth_label: str, lenience: float) -> OutputPaths:
    """Derive destination paths using the depth and model labels."""
    model_slug = slugify(model_name)
    lenience_slug = slugify(f"{lenience:.2f}")
    prefix = f"competency_llm_depth-{depth_label}_model-{model_slug}_lenience-{lenience_slug}"
    cache = REPORT_DIR / f"{prefix}_cache.json"
    summary = REPORT_DIR / f"{prefix}_results.json"
    report = REPORT_DIR / f"{prefix}_report.md"
    profile_summary = REPORT_DIR / f"{prefix}_profile.json"
    profile_dir = REPORT_DIR / f"{prefix}_subgraph-profiles"
    return OutputPaths(
        cache=cache,
        summary=summary,
        report=report,
        profile_summary=profile_summary,
        profile_dir=profile_dir,
    )


def classify_term(term: object) -> str:
    """Map an RDFLib term instance onto a coarse string label for monitoring.

    Args:
        term: RDFLib term to classify.

    Returns:
        str: Coarse-grained label describing the term type.
    """
    if isinstance(term, URIRef):
        return "uri"
    if isinstance(term, Literal):
        return "literal"
    if isinstance(term, BNode):
        return "bnode"
    return term.__class__.__name__.lower()


def format_subgraph_query(seeds: set[URIRef], depth: int | None, mode_label: str) -> str:
    """Render neighbourhood parameters into a readable query string."""
    if mode_label == "full":
        return "full_graph_serialisation"
    if seeds:
        seeds_text = ", ".join(sorted(str(seed) for seed in seeds))
    else:
        seeds_text = "UNSEEDED (full union graph)"
    depth_text = "None" if depth is None else depth
    return f"expand_subgraph(depth={depth_text}, seeds=[{seeds_text}])"


def summarise_subgraph(
    seeds: set[URIRef],
    depth: int | None,
    subgraph: Graph,
    mode_label: str,
) -> SubgraphMonitoring:
    """Produce monitoring metadata describing the retrieved subgraph.

    Args:
        seeds: Seeds used to anchor the subgraph expansion.
        depth: Hop depth used during expansion.
        subgraph: Resulting subgraph.
        mode_label: Evaluation mode identifier ("subgraph" or "full").

    Returns:
        SubgraphMonitoring: Observability snapshot for persistence and reporting.
    """
    predicate_counts = Counter(str(predicate) for _, predicate, _ in subgraph)
    top_predicates = predicate_counts.most_common(10)
    predicate_breakdown: list[dict[str, int]] = [
        {"predicate": predicate, "count": count} for predicate, count in top_predicates
    ]
    remainder = len(subgraph) - sum(item["count"] for item in predicate_breakdown)
    if remainder > 0:
        predicate_breakdown.append({"predicate": "__other__", "count": remainder})

    subject_term_types = Counter(classify_term(subject) for subject, _, _ in subgraph)
    object_term_types = Counter(classify_term(obj) for _, _, obj in subgraph)

    return SubgraphMonitoring(
        retrieval_query=format_subgraph_query(seeds, depth, mode_label),
        depth=depth,
        seeds=sorted(str(seed) for seed in seeds),
        triple_count=len(subgraph),
        predicate_breakdown=predicate_breakdown,
        subject_term_types=dict(sorted(subject_term_types.items())),
        object_term_types=dict(sorted(object_term_types.items())),
    )


class ValueNormalizer:
    """Convert predicted strings into canonical identifiers (URIs when possible)."""

    def __init__(self, dataset: ConjunctiveGraph):
        graph = dataset
        self.ns_manager = graph.namespace_manager

        label_candidates: dict[str, set[str]] = {}
        local_candidates: dict[str, set[str]] = {}

        for subject, _, label in graph.triples((None, RDFS.label, None)):
            if not isinstance(subject, URIRef):
                continue
            key = self._normalise_key(str(label))
            if not key:
                continue
            label_candidates.setdefault(key, set()).add(str(subject))

        for subject in set(graph.subjects()):
            if not isinstance(subject, URIRef):
                continue
            local = self._extract_local_name(str(subject))
            if not local:
                continue
            key = local.lower()
            local_candidates.setdefault(key, set()).add(str(subject))

        self.label_map = {k: next(iter(v)) for k, v in label_candidates.items() if len(v) == 1}
        self.local_map = {k: next(iter(v)) for k, v in local_candidates.items() if len(v) == 1}

    @staticmethod
    def _normalise_key(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).lower()

    @staticmethod
    def _extract_local_name(iri: str) -> str | None:
        if "#" in iri:
            return iri.rsplit("#", 1)[1]
        if "/" in iri:
            return iri.rstrip("/").rsplit("/", 1)[-1]
        return None

    def canonical(self, value: str) -> str:
        v = value.strip()
        if not v:
            return ""
        if v.startswith("<") and v.endswith(">"):
            v = v[1:-1]
        if v.lower().startswith(("http://", "https://")):
            return str(v)
        if ":" in v:
            prefix, local = v.split(":", 1)
            namespace = self.ns_manager.store.namespace(prefix)
            if namespace is not None:
                return str(namespace + local)
        key = self._normalise_key(v)
        uri = self.label_map.get(key)
        if uri:
            return uri
        uri = self.local_map.get(key)
        if uri:
            return uri
        return key


def load_dataset() -> ConjunctiveGraph:
    """Parse the generated RDF artefacts into a conjunctive graph."""
    dataset = ConjunctiveGraph()
    for path in sorted(GENERATED_RDF_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix not in {".ttl", ".trig"}:
            continue
        if path.name.startswith(EXCLUDED_PREFIXES):
            continue
        fmt = "turtle" if suffix == ".ttl" else "trig"
        dataset.parse(str(path), format=fmt)
    return dataset


def load_queries(paths: Iterable[Path]) -> dict[Path, str]:
    """Read the text of competency query files."""
    return {path: path.read_text(encoding="utf-8") for path in paths}


def extract_question_text(path: Path) -> str:
    """Extract the first comment line as the question text."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    raise ValueError(f"No question comment found in {path}")


def extract_seeds(query_text: str) -> set[URIRef]:
    """Extract namespace-prefixed tokens from the query as seed IRIs."""
    seeds: set[URIRef] = set()
    for match in re.findall(r":([A-Za-z0-9_]+)", query_text):
        seeds.add(BASE_NS[match])
    return seeds


def infer_literal_seeds(dataset: ConjunctiveGraph, query_text: str) -> set[URIRef]:
    """Infer additional seed IRIs by matching literal filters against the dataset.

    Args:
        dataset: Graph to search for literal bindings.
        query_text: SPARQL query text.

    Returns:
        set[URIRef]: Subjects whose properties match literal constraints in the query.
    """
    literal_seed_candidates: set[URIRef] = set()
    for predicate_local, literal_value in re.findall(r":([A-Za-z0-9_]+)\s+\"([^\"]+)\"", query_text):
        predicate = BASE_NS[predicate_local]
        for subject in dataset.subjects(predicate, Literal(literal_value)):
            if isinstance(subject, URIRef):
                literal_seed_candidates.add(subject)
    return literal_seed_candidates


def expand_subgraph(dataset: ConjunctiveGraph, seeds: set[URIRef], depth: int) -> Graph:
    """Expand a neighbourhood around the seeds using BFS over the union graph."""
    if not seeds:
        LOGGER.warning("No seeds detected; using full dataset union graph")
        subgraph = Graph()
        subgraph.namespace_manager = dataset.namespace_manager
        for triple in dataset:
            subgraph.add(triple)
        return subgraph

    union_graph = dataset
    subgraph = Graph()
    subgraph.namespace_manager = union_graph.namespace_manager

    visited: set[URIRef] = set()
    queue: deque[tuple[URIRef, int]] = deque((seed, 0) for seed in seeds)

    while queue:
        node, dist = queue.popleft()
        if node in visited or not isinstance(node, URIRef):
            continue
        visited.add(node)

        for s, p, o in union_graph.triples((node, None, None)):
            subgraph.add((s, p, o))
            if isinstance(o, URIRef) and dist < depth:
                queue.append((o, dist + 1))
        for s, p, o in union_graph.triples((None, None, node)):
            subgraph.add((s, p, o))
            if isinstance(s, URIRef) and dist < depth:
                queue.append((s, dist + 1))

    return subgraph


def run_sparql(dataset: ConjunctiveGraph, query: str) -> tuple[list[str], list[tuple[str, ...]]]:
    """Execute a SPARQL query against the dataset and stringify the bindings."""
    result = dataset.query(query)
    columns = [str(var) for var in result.vars]
    rows: list[tuple[str, ...]] = []
    for record in result:
        row = tuple("" if value is None else str(value) for value in record)
        rows.append(row)
    return columns, rows


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from an LLM response payload."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 2 else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


def canonicalise_rows(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    normaliser: ValueNormalizer,
) -> tuple[list[tuple[str, ...]], set[tuple[str, ...]]]:
    """Canonicalise row values using the supplied normaliser."""
    canonical_rows: list[tuple[str, ...]] = []
    for row in rows:
        canonical_rows.append(tuple(normaliser.canonical(str(value)) for value in row[: len(columns)]))
    return canonical_rows, set(canonical_rows)


def rows_match(
    gold_row: Sequence[str],
    pred_row: Sequence[str],
    max_mismatches: int,
) -> bool:
    """Determine whether two rows match under the mismatch budget."""
    mismatches = sum(1 for gold_cell, pred_cell in zip(gold_row, pred_row) if gold_cell != pred_cell)
    return mismatches <= max_mismatches


def score_predictions(
    gold_rows: Sequence[tuple[str, ...]],
    pred_rows: Sequence[tuple[str, ...]],
    max_mismatches: int,
) -> tuple[float, float, float, int, int, int]:
    """Calculate metrics allowing per-row mismatches within the lenience budget."""
    if not gold_rows and not pred_rows:
        return 1.0, 1.0, 1.0, 0, 0, 0
    gold_used = [False] * len(gold_rows)
    matches = 0
    for pred in pred_rows:
        for idx, gold in enumerate(gold_rows):
            if gold_used[idx]:
                continue
            if rows_match(gold, pred, max_mismatches):
                gold_used[idx] = True
                matches += 1
                break

    tp = matches
    fp = len(pred_rows) - matches
    fn = len(gold_rows) - matches

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def call_llm(
    client: OpenAI,
    model_name: str,
    question: str,
    context_text: str,
    expected_columns: Sequence[str],
) -> dict:
    """Invoke the chat completion API with the extracted context."""
    user_prompt = (
        "Context (documentation subgraph in Turtle):\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        f"- Return ONLY JSON with two keys: \"columns\" and \"rows\".\n"
        f"- The \"columns\" array must exactly be {list(expected_columns)}.\n"
        "- The \"rows\" array must contain an array of IRIs per answer, aligned to the column order.\n"
        "- Every resource MUST be written as the exact IRI present in the context (full IRI preferred; CURIE allowed only if defined).\n"
        "- Use empty strings for unknown cells, and an empty array for \"rows\" if the context lacks evidence.\n"
        "- Do not include explanations or additional keys.\n"
        'Example: {"columns": ["https://example.org/a", "https://example.org/b"], "rows": [["https://example.org/x", "https://example.org/y"]]}'
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer maintenance ontology questions using only the provided RDF context. "
                "Respect the requested JSON schema."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(model=model_name, messages=messages)
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(strip_code_fences(content))
    except json.JSONDecodeError as exc:
        LOGGER.warning(f"Model response was not valid JSON: {content}")
        return {}


def compute_context_hash(context_text: str) -> str:
    """Hash the serialised context for cache key construction."""
    return hashlib.sha256(context_text.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> dict[str, dict]:
    """Load the cache JSON from the provided path."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Cache at %s is invalid JSON; starting fresh", cache_path)
    return {}


def save_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    """Persist the cache JSON to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def make_cache_key(
    path: Path,
    question: str,
    query_text: str,
    context_hash: str,
    expected_columns: Sequence[str],
    depth_label: str,
    model_name: str,
) -> str:
    """Construct a deterministic cache key for the current question/context."""
    payload = {
        "path": str(path),
        "question": question,
        "query": query_text,
        "context_hash": context_hash,
        "columns": list(expected_columns),
        "depth": depth_label,
        "model": model_name,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fetch_cache_entry(cache: dict[str, dict], cache_key: str) -> dict | None:
    """Return the cached response for the given key, if present."""
    entry = cache.get(cache_key)
    if entry is None:
        return None
    if isinstance(entry, dict) and "_meta" in entry:
        return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}
    return entry


def fallback_cache_entry(
    cache: dict[str, dict],
    path: Path,
    expected_columns: Sequence[str],
    gold_size: int,
    legacy_paths: Sequence[Path],
) -> dict | None:
    """Search for a reusable cache entry across in-memory and legacy caches."""

    def _scan(entries: dict[str, dict]) -> dict | None:
        target_columns = list(expected_columns)
        candidates: list[dict] = []
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            meta = entry.get("_meta")
            if isinstance(meta, dict) and meta.get("path") == str(path):
                return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}
            if entry.get("columns") == target_columns:
                candidates.append(entry)
        if not candidates:
            return None
        if len(candidates) == 1:
            return {"columns": candidates[0].get("columns", []), "rows": candidates[0].get("rows", [])}
        for entry in candidates:
            if len(entry.get("rows", [])) == gold_size:
                return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}
        entry = candidates[0]
        return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}

    hit = _scan(cache)
    if hit:
        return hit
    for legacy_path in legacy_paths:
        if legacy_path.exists():
            try:
                external_cache = json.loads(legacy_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                external_cache = {}
            hit = _scan(external_cache)
            if hit:
                return hit
    return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate documentation competency questions using LLM outputs against gold SPARQL answers."
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Reuse cached LLM responses and skip API calls. Questions without cached responses are skipped.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_SUBGRAPH_DEPTH,
        help="Depth (in hops) for subgraph expansion (default: %(default)s). Ignored when --full-graph is set.",
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Use the entire union graph as context (legacy mode). Overrides --depth.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name to invoke (default: %(default)s).",
    )
    parser.add_argument(
        "--lenience",
        type=float,
        default=DEFAULT_LENIENCE,
        help=(
            "Scoring lenience in the [0, 1] range. A value of 0.0 requires exact row matches, "
            "while higher values allow proportionally more mismatched cells per row (e.g. 0.25 "
            "permits a single mismatch in four-column rows)."
        ),
    )
    parser.add_argument(
        "--profile-subgraphs",
        action="store_true",
        help=(
            "Collect subgraph telemetry without calling the LLM. "
            "Writes Turtle snapshots under the model/depth-specific profile directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for CLI usage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    use_full_graph = args.full_graph
    model_name = args.model
    depth_value = None if use_full_graph else args.depth
    depth_label = "full" if use_full_graph else str(depth_value)
    lenience = args.lenience
    if not 0.0 <= lenience <= 1.0:
        raise ValueError(f"Lenience must be within [0, 1]; received {lenience}")
    output_paths = build_output_paths(model_name, depth_label, lenience)

    dataset = load_dataset()
    documentation_queries = sorted(DOCUMENTATION_QUERY_DIR.glob("*.rq"))
    queries = load_queries(documentation_queries)

    cache = {} if args.profile_subgraphs else load_cache(output_paths.cache)
    normaliser = ValueNormalizer(dataset)
    cache_modified = False

    client = None if args.profile_subgraphs else OpenAI(api_key=API_KEY_PATH.read_text(encoding="utf-8").strip())
    metrics: list[dict] = []
    skipped_due_to_cache: list[str] = []
    profile_entries: list[dict[str, object]] = []
    mode_label = "full" if use_full_graph else "subgraph"

    full_context_text: str | None = None
    full_context_hash: str | None = None
    full_monitoring_dict: dict[str, object] | None = None
    if use_full_graph:
        serialized = dataset.serialize(format="turtle")
        if isinstance(serialized, bytes):
            serialized = serialized.decode("utf-8")
        full_context_text = serialized
        full_context_hash = compute_context_hash(serialized)
        full_monitoring_dict = summarise_subgraph(set(), None, dataset, mode_label).asdict()

    for path, query_text in queries.items():
        question = extract_question_text(path)
        columns, gold_rows = run_sparql(dataset, query_text)
        gold_rows_list, gold_set = canonicalise_rows(columns, gold_rows, normaliser)
        gold_rows_canonical = [list(row) for row in sorted(gold_set)]

        if use_full_graph:
            monitoring_dict = full_monitoring_dict or {}
            context_text = full_context_text or ""
            context_hash = full_context_hash or compute_context_hash(context_text)
        else:
            seeds = extract_seeds(query_text)
            literal_seeds = infer_literal_seeds(dataset, query_text)
            if literal_seeds:
                LOGGER.info(
                    "Augmented seeds for %s with %d literal matches",
                    path.name,
                    len(literal_seeds),
                )
                seeds.update(literal_seeds)
            subgraph = expand_subgraph(dataset, seeds, depth_value or DEFAULT_SUBGRAPH_DEPTH)
            monitoring_dict = summarise_subgraph(seeds, depth_value or DEFAULT_SUBGRAPH_DEPTH, subgraph, mode_label).asdict()
            LOGGER.info(
                "Subgraph retrieval for %s retrieved %d triples (depth=%s, seeds=%d)",
                path.name,
                monitoring_dict["triple_count"],
                depth_label,
                len(monitoring_dict["seeds"]),
            )
            context_text = subgraph.serialize(format="turtle")
            if isinstance(context_text, bytes):
                context_text = context_text.decode("utf-8")
            context_hash = compute_context_hash(context_text)

        max_mismatches = math.floor(len(columns) * lenience)

        if args.profile_subgraphs:
            output_paths.profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = output_paths.profile_dir / f"{path.stem}.ttl"
            profile_path.write_text(context_text, encoding="utf-8")
            if monitoring_dict.get("triple_count", 0) == 0:
                LOGGER.warning(
                    "Profiling: %s produced 0 triples. Seeds=%s",
                    path.name,
                    ", ".join(monitoring_dict.get("seeds", [])) or "(none)",
                )
            profile_entries.append(
                {
                    "question": question,
                    "query_path": str(path),
                    "mode": mode_label,
                    "depth": depth_label,
                    "model": model_name,
                    "lenience": lenience,
                    "max_mismatches": max_mismatches,
                    "subgraph_monitoring": monitoring_dict,
                    "subgraph_path": str(profile_path),
                    "context_hash": context_hash,
                }
            )
            continue

        cache_key = make_cache_key(path, question, query_text, context_hash, columns, depth_label, model_name)
        cache_entry = cache.get(cache_key)
        model_json = fetch_cache_entry(cache, cache_key)
        if model_json is None and args.cache_only:
            model_json = fallback_cache_entry(cache, path, columns, len(gold_rows_canonical), LEGACY_CACHE_PATHS)
            if model_json:
                LOGGER.info(
                    "Using cached response for %s (%s) via fallback",
                    path.name,
                    question,
                )
                cache_entry = cache.setdefault(
                    cache_key,
                    {"columns": model_json.get("columns", []), "rows": model_json.get("rows", [])},
                )
                cache_modified = True
        if model_json is None:
            if args.cache_only:
                LOGGER.warning(
                    "Cache miss for %s (%s) under --cache-only; skipping question.",
                    path.name,
                    question,
                )
                skipped_due_to_cache.append(question)
                continue
            if client is None:
                raise RuntimeError("LLM client not initialised; cannot query model outside of --cache-only or profiling mode.")
            LOGGER.info("Querying LLM (%s) for %s (%s)", model_name, path.name, question)
            model_json = call_llm(client, model_name, question, context_text, columns)
            cache_entry = {
                "columns": model_json.get("columns", []),
                "rows": model_json.get("rows", []),
            }
            cache[cache_key] = cache_entry
            cache_modified = True

        if cache_entry is None:
            cache_entry = cache.setdefault(
                cache_key,
                {"columns": model_json.get("columns", []), "rows": model_json.get("rows", [])},
            )
            cache_modified = True
        else:
            cache_entry["columns"] = model_json.get("columns", [])
            cache_entry["rows"] = model_json.get("rows", [])
            cache_modified = True

        meta = cache_entry.setdefault("_meta", {})
        meta.update(
            {
                "path": str(path),
                "question": question,
                "columns": list(columns),
                "context_hash": context_hash,
                "depth": depth_label,
                "model": model_name,
                "mode": mode_label,
                "lenience": lenience,
                "max_mismatches": max_mismatches,
                "subgraph_monitoring": monitoring_dict,
            }
        )

        model_columns = [str(col) for col in model_json.get("columns", [])]
        model_rows_raw = model_json.get("rows", [])
        if not isinstance(model_rows_raw, list):
            raise ValueError(f"Model rows must be a list: {model_json}")
        model_rows = [
            [str(cell) for cell in row] if isinstance(row, (list, tuple)) else [str(row)]
            for row in model_rows_raw
        ]

        pred_rows_list: list[tuple[str, ...]] = []
        canonical_pred_rows, pred_set = ([], set())
        if model_columns and len(model_columns) == len(columns):
            pred_rows_list, pred_set = canonicalise_rows(columns, model_rows, normaliser)
            canonical_pred_rows = [list(row) for row in sorted(pred_set)]
        else:
            canonical_pred_rows = []
        precision, recall, f1, tp, fp, fn = score_predictions(gold_rows_list, pred_rows_list, max_mismatches)
        metrics.append(
            {
                "path": path,
                "question": question,
                "columns": columns,
                "gold_rows": gold_rows_canonical,
                "pred_rows": canonical_pred_rows,
                "pred_rows_raw": model_rows,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "context_hash": context_hash,
                "max_mismatches": max_mismatches,
                "subgraph_monitoring": monitoring_dict,
            }
        )

    if args.profile_subgraphs:
        output_paths.profile_summary.parent.mkdir(parents=True, exist_ok=True)
        profile_summary = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "model": model_name,
            "mode": mode_label,
            "subgraph_depth": depth_label,
            "lenience": lenience,
            "profiles": profile_entries,
        }
        output_paths.profile_summary.write_text(json.dumps(profile_summary, indent=2, sort_keys=True), encoding="utf-8")
        for entry in profile_entries:
            monitoring = entry["subgraph_monitoring"]
            print(
                f"{Path(entry['query_path']).name}: triples={monitoring['triple_count']} "
                f"seeds={len(monitoring['seeds'])} depth={depth_label}"
            )
        print(f"\nProfiling summaries written to {output_paths.profile_summary}")
        return

    if cache_modified:
        save_cache(output_paths.cache, cache)
        LOGGER.info("Updated cache written to %s", output_paths.cache)
    else:
        LOGGER.info("LLM response cache stored at %s", output_paths.cache)

    macro_precision = statistics.mean(m["precision"] for m in metrics) if metrics else math.nan
    macro_recall = statistics.mean(m["recall"] for m in metrics) if metrics else math.nan
    macro_f1 = statistics.mean(m["f1"] for m in metrics) if metrics else math.nan

    total_tp = sum(m["tp"] for m in metrics)
    total_fp = sum(m["fp"] for m in metrics)
    total_fn = sum(m["fn"] for m in metrics)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    if skipped_due_to_cache:
        LOGGER.info(
            "Skipped %d questions due to cache-only mode: %s",
            len(skipped_due_to_cache),
            ", ".join(skipped_due_to_cache),
        )
    for entry in metrics:
        print(
            f"{entry['path'].name}: P={entry['precision']:.2f} "
            f"R={entry['recall']:.2f} F1={entry['f1']:.2f} "
            f"(gold={len(entry['gold_rows'])}, pred={len(entry['pred_rows'])})"
        )
    print("\nMicro averages")
    print("--------------")
    print(f"Precision: {micro_precision:.2f}")
    print(f"Recall:    {micro_recall:.2f}")
    print(f"F1:        {micro_f1:.2f}")
    print("\nMacro averages")
    print("--------------")
    print(f"Precision: {macro_precision:.2f}")
    print(f"Recall:    {macro_recall:.2f}")
    print(f"F1:        {macro_f1:.2f}")

    report_lines = [
        "# Competency LLM Evaluation",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Mode: {mode_label}",
        f"- Subgraph depth: {depth_label}",
        f"- Model: {model_name}",
        f"- Lenience: {lenience:.2f} (fractional mismatch budget per row)",
        "",
        "## Macro Metrics",
        "",
        f"- Precision: {macro_precision:.4f}",
        f"- Recall: {macro_recall:.4f}",
        f"- F1: {macro_f1:.4f}",
        "",
        "## Micro Metrics",
        "",
        f"- Precision: {micro_precision:.4f}",
        f"- Recall: {micro_recall:.4f}",
        f"- F1: {micro_f1:.4f}",
        "",
        "## Per-Question Results",
        "",
        "| Question | Precision | Recall | F1 | Gold Rows | Pred Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in metrics:
        report_lines.append(
            f"| {entry['question']} | {entry['precision']:.2f} | {entry['recall']:.2f} | "
            f"{entry['f1']:.2f} | {len(entry['gold_rows'])} | {len(entry['pred_rows'])} |"
        )
    report_lines.extend(
        [
            "",
            "## Subgraph Retrieval Monitoring",
            "",
            "| Question | Retrieval Query | Triples | Term Types | Top Predicates | Mismatch Budget |",
            "| --- | --- | ---: | --- | --- | ---: |",
        ]
    )
    for entry in metrics:
        monitoring = entry["subgraph_monitoring"]
        retrieval_query = monitoring.get("retrieval_query", "")
        subject_types = monitoring.get("subject_term_types", {})
        object_types = monitoring.get("object_term_types", {})
        subject_summary = ", ".join(f"{key}:{value}" for key, value in subject_types.items()) or "—"
        object_summary = ", ".join(f"{key}:{value}" for key, value in object_types.items()) or "—"
        term_summary = f"subj {subject_summary}; obj {object_summary}"
        predicate_breakdown = monitoring.get("predicate_breakdown", [])
        top_predicates = [
            f"{item.get('predicate', '')} ({item.get('count', 0)})"
            for item in predicate_breakdown[:3]
        ]
        if len(predicate_breakdown) > 3:
            top_predicates.append(f"+{len(predicate_breakdown) - 3} more")
        predicate_summary = "; ".join(top_predicates) or "—"
        report_lines.append(
            f"| {entry['question']} | `{retrieval_query}` | {monitoring.get('triple_count', 0)} | "
            f"{term_summary} | {predicate_summary} | {entry['max_mismatches']} |"
        )
    output_paths.report.parent.mkdir(parents=True, exist_ok=True)
    output_paths.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": mode_label,
        "model": model_name,
        "subgraph_depth": depth_label,
        "lenience": lenience,
        "skipped_questions": skipped_due_to_cache,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "results": [
            {
                "question": entry["question"],
                "query_path": str(entry["path"]),
                "columns": entry["columns"],
                "gold_rows": entry["gold_rows"],
                "pred_rows": entry["pred_rows"],
                "pred_rows_raw": entry["pred_rows_raw"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1": entry["f1"],
                "tp": entry["tp"],
                "fp": entry["fp"],
                "fn": entry["fn"],
                "context_hash": entry["context_hash"],
                "max_mismatches": entry["max_mismatches"],
                "subgraph_monitoring": entry["subgraph_monitoring"],
            }
            for entry in metrics
        ],
    }
    output_paths.summary.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
