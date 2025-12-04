"""Evaluate documentation competency questions with optional subgraph contexts.

Reproducibility features:
- Stable RDF context serialisation (canonical-ish N-Triples + sorted lines).
- Fixed sampling controls: temperature, top_p, seed (best-effort determinism).
- Cache keys include sampling params + context hash.
- Multiple runs (typically with different seeds) with mean/stddev of macro/micro metrics.

Notes:
- Even with seed + temperature=0, determinism is best-effort; record system_fingerprint to detect backend changes.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Sequence, Any
from decimal import Decimal, InvalidOperation
from functools import lru_cache

import requests

from openai import OpenAI
from rdflib import ConjunctiveGraph, Graph, Namespace, URIRef
from rdflib.compare import to_isomorphic
from rdflib.namespace import RDFS
from rdflib.term import BNode, Literal


# Pinned dated model snapshot (override with --model if desired).
DEFAULT_MODEL = "gpt-5-nano-2025-08-07"

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

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_RUNS = 1
DEFAULT_SEED_STEP = 1

LOGGER = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class OutputPaths:
    cache: Path
    summary: Path
    report: Path
    profile_summary: Path
    profile_dir: Path
    cost_estimate: Path


@dataclass(frozen=True)
class SubgraphMonitoring:
    retrieval_query: str
    depth: int | None
    seeds: list[str]
    triple_count: int
    predicate_breakdown: list[dict[str, int]]
    subject_term_types: dict[str, int]
    object_term_types: dict[str, int]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionContext:
    path: Path
    question: str
    query_text: str
    columns: list[str]
    gold_rows_list: list[tuple[str, ...]]
    gold_rows_canonical: list[list[str]]
    max_mismatches: int
    context_text: str
    context_hash: str
    monitoring_dict: dict[str, object]


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "value"


def build_output_paths(
    model_name: str,
    depth_label: str,
    lenience: float,
    temperature: float,
    top_p: float,
    base_seed: int,
    runs: int,
    seed_step: int,
) -> OutputPaths:
    model_slug = slugify(model_name)
    lenience_slug = slugify(f"{lenience:.2f}")
    temp_slug = slugify(f"{temperature:.3f}")
    top_p_slug = slugify(f"{top_p:.3f}")
    seed_slug = str(base_seed)
    runs_slug = str(runs)
    step_slug = str(seed_step)

    prefix = (
        f"competency_llm_depth-{depth_label}"
        f"_model-{model_slug}"
        f"_lenience-{lenience_slug}"
        f"_temp-{temp_slug}"
        f"_top_p-{top_p_slug}"
        f"_seed-{seed_slug}"
        f"_runs-{runs_slug}"
        f"_seedstep-{step_slug}"
    )
    cache = REPORT_DIR / f"{prefix}_cache.json"
    summary = REPORT_DIR / f"{prefix}_results.json"
    report = REPORT_DIR / f"{prefix}_report.md"
    profile_summary = REPORT_DIR / f"{prefix}_profile.json"
    profile_dir = REPORT_DIR / f"{prefix}_subgraph-profiles"
    cost_estimate = REPORT_DIR / f"{prefix}_cost-estimate.json"
    return OutputPaths(
        cache=cache,
        summary=summary,
        report=report,
        profile_summary=profile_summary,
        profile_dir=profile_dir,
        cost_estimate=cost_estimate,
    )


def classify_term(term: object) -> str:
    if isinstance(term, URIRef):
        return "uri"
    if isinstance(term, Literal):
        return "literal"
    if isinstance(term, BNode):
        return "bnode"
    return term.__class__.__name__.lower()


def format_subgraph_query(seeds: set[URIRef], depth: int | None, mode_label: str) -> str:
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
    return {path: path.read_text(encoding="utf-8") for path in paths}


def extract_question_text(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    raise ValueError(f"No question comment found in {path}")


def extract_seeds(query_text: str) -> set[URIRef]:
    seeds: set[URIRef] = set()
    for match in re.findall(r":([A-Za-z0-9_]+)", query_text):
        seeds.add(BASE_NS[match])
    return seeds


def infer_literal_seeds(dataset: ConjunctiveGraph, query_text: str) -> set[URIRef]:
    literal_seed_candidates: set[URIRef] = set()
    for predicate_local, literal_value in re.findall(r":([A-Za-z0-9_]+)\s+\"([^\"]+)\"", query_text):
        predicate = BASE_NS[predicate_local]
        for subject in dataset.subjects(predicate, Literal(literal_value)):
            if isinstance(subject, URIRef):
                literal_seed_candidates.add(subject)
    return literal_seed_candidates


def expand_subgraph(dataset: ConjunctiveGraph, seeds: set[URIRef], depth: int) -> Graph:
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
    result = dataset.query(query)
    columns = [str(var) for var in result.vars]
    rows: list[tuple[str, ...]] = []
    for record in result:
        row = tuple("" if value is None else str(value) for value in record)
        rows.append(row)
    return columns, rows


def strip_code_fences(text: str) -> str:
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
    canonical_rows: list[tuple[str, ...]] = []
    for row in rows:
        canonical_rows.append(tuple(normaliser.canonical(str(value)) for value in row[: len(columns)]))
    return canonical_rows, set(canonical_rows)


def rows_match(gold_row: Sequence[str], pred_row: Sequence[str], max_mismatches: int) -> bool:
    mismatches = sum(1 for gold_cell, pred_cell in zip(gold_row, pred_row) if gold_cell != pred_cell)
    return mismatches <= max_mismatches


def score_predictions(
    gold_rows: Sequence[tuple[str, ...]],
    pred_rows: Sequence[tuple[str, ...]],
    max_mismatches: int,
) -> tuple[float, float, float, int, int, int]:
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


def stable_rdf_text(graph: Any) -> str:
    """Deterministic-ish RDF text for hashing + prompting.
    - Canonicalises blank nodes via isomorphism.
    - Serialises to N-Triples and sorts lines.
    """
    iso = to_isomorphic(graph)
    nt = iso.serialize(format="nt")
    if isinstance(nt, bytes):
        nt = nt.decode("utf-8")
    lines = sorted(line for line in nt.splitlines() if line.strip())
    return "\n".join(lines) + "\n"


def compute_context_hash(context_text: str) -> str:
    return hashlib.sha256(context_text.encode("utf-8")).hexdigest()


def read_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"Empty API key file: {path}")
    return key


def is_openrouter_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "openrouter.ai" in base_url.lower()


def build_messages(question: str, context_text: str, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    # Deterministic rendering of expected columns.
    columns_json = json.dumps(list(expected_columns), ensure_ascii=False)
    user_prompt = (
        "Context (documentation subgraph as sorted N-Triples):\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        '- Return ONLY JSON with two keys: "columns" and "rows".\n'
        f'- The "columns" array must exactly be {columns_json}.\n'
        '- The "rows" array must contain an array of strings per answer, aligned to the column order.\n'
        "- Every resource MUST be written as the exact IRI present in the context (full IRI preferred; CURIE allowed only if defined).\n"
        '- Use empty strings for unknown cells, and an empty array for "rows" if the context lacks evidence.\n'
        "- Do not include explanations or additional keys.\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You answer maintenance ontology questions using only the provided RDF context. "
                "Respect the requested JSON schema."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


@lru_cache(maxsize=1)
def _openrouter_model_map_cached(api_key: str) -> dict[str, dict]:
    # Cached for determinism within one run (pricing snapshot fetched once).
    r = requests.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    if not isinstance(data, list):
        return {}
    out: dict[str, dict] = {}
    for m in data:
        if isinstance(m, dict) and isinstance(m.get("id"), str):
            out[m["id"]] = m
    return out


def _decimal_or_zero(value: object) -> Decimal:
    try:
        if value is None:
            return Decimal("0")
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def get_openrouter_prompt_pricing(api_key: str, model_id: str) -> dict[str, str] | None:
    # Returns prompt + request pricing as strings for stable JSON output.
    model_map = _openrouter_model_map_cached(api_key)
    m = model_map.get(model_id)
    if not isinstance(m, dict):
        return None
    pricing = m.get("pricing")
    if not isinstance(pricing, dict):
        return None
    prompt = _decimal_or_zero(pricing.get("prompt"))
    request = _decimal_or_zero(pricing.get("request"))
    return {
        "source": "openrouter_models_api",
        "model_id": model_id,
        "prompt_usd_per_token": format(prompt, "f"),
        "request_usd": format(request, "f"),
    }


def count_prompt_tokens(
    *,
    messages: list[dict[str, str]],
    model_name: str,
    base_url: str | None,
) -> tuple[int, str]:
    """
    Deterministic token count.
    - Preferred: LiteLLM token_counter (best match for many providers).
    - Fallback: tiktoken for OpenAI-family models.
    - Last resort: deterministic approximation (chars//4).
    """
    # 1) LiteLLM (preferred)
    try:
        from litellm import token_counter  # type: ignore

        litellm_model = model_name
        if is_openrouter_base_url(base_url):
            # LiteLLM expects openrouter/<id> for OpenRouter-hosted model IDs.
            litellm_model = f"openrouter/{model_name}"
        tokens = int(token_counter(model=litellm_model, messages=messages))
        return tokens, "litellm_token_counter"
    except Exception:
        pass

    # 2) tiktoken (best-effort)
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.encoding_for_model(model_name)
        # This is an approximation for chat-format models; still deterministic.
        text = "".join(f"{m.get('role','')}:{m.get('content','')}\n" for m in messages)
        return len(enc.encode(text)), "tiktoken_concat_fallback"
    except Exception:
        pass

    # 3) Deterministic coarse approximation
    text = "".join(f"{m.get('role','')}:{m.get('content','')}\n" for m in messages)
    return max(1, len(text) // 4), "chars_div_4_fallback"


def estimate_input_cost(
    *,
    messages: list[dict[str, str]],
    model_name: str,
    base_url: str | None,
    pricing: dict[str, str] | None,
) -> dict[str, object]:
    tokens, method = count_prompt_tokens(messages=messages, model_name=model_name, base_url=base_url)
    out: dict[str, object] = {
        "prompt_tokens_est": tokens,
        "token_count_method": method,
    }
    if pricing is None:
        out["pricing"] = None
        out["input_cost_est_usd"] = None
        return out

    prompt = _decimal_or_zero(pricing.get("prompt_usd_per_token"))
    request = _decimal_or_zero(pricing.get("request_usd"))
    cost = (prompt * Decimal(tokens)) + request
    out["pricing"] = pricing
    out["input_cost_est_usd"] = format(cost, "f")
    return out


def load_cache(cache_path: Path) -> dict[str, dict]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Cache at %s is invalid JSON; starting fresh", cache_path)
    return {}


def save_cache(cache_path: Path, cache: dict[str, dict]) -> None:
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
    *,
    temperature: float,
    top_p: float,
    seed: int,
    response_format: str,
) -> str:
    payload = {
        "path": str(path),
        "question": question,
        "query": query_text,
        "context_hash": context_hash,
        "columns": list(expected_columns),
        "depth": depth_label,
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "response_format": response_format,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def get_cached_payload(cache: dict[str, dict], cache_key: str) -> dict | None:
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    # Accept both new and legacy layouts.
    columns = entry.get("columns", [])
    rows = entry.get("rows", [])
    meta = entry.get("_meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return {"columns": columns, "rows": rows, "_meta": meta}


def fallback_cache_entry(
    cache: dict[str, dict],
    path: Path,
    expected_columns: Sequence[str],
    gold_size: int,
    legacy_paths: Sequence[Path],
) -> dict | None:
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


def call_llm(
    client: OpenAI,
    model_name: str,
    question: str,
    context_text: str,
    expected_columns: Sequence[str],
    *,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[dict, dict]:
    messages = build_messages(question, context_text, expected_columns)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"

    # system_fingerprint may be absent depending on endpoint/model.
    system_fingerprint = getattr(response, "system_fingerprint", None)

    meta = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "response_format": "json_object",
        "system_fingerprint": system_fingerprint,
    }

    try:
        return json.loads(strip_code_fences(content)), meta
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON: {content}") from exc


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))


def parse_args() -> argparse.Namespace:
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
        help="OpenAI model name to invoke (default: pinned dated snapshot).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Optional API base URL for the OpenAI SDK (e.g. https://openrouter.ai/api/v1). "
            "If this points at OpenRouter, --estimate-input-cost will use OpenRouter model pricing."
        ),
    )
    parser.add_argument(
        "--lenience",
        type=float,
        default=DEFAULT_LENIENCE,
        help="Scoring lenience in the [0, 1] range (fractional mismatch budget per row).",
    )
    parser.add_argument(
        "--profile-subgraphs",
        action="store_true",
        help=(
            "Collect subgraph telemetry without calling the LLM. "
            "Writes stable N-Triples snapshots under the model/depth-specific profile directory."
        ),
    )

    # Reproducibility controls.
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for run 0. Run i uses seed + i*seed_step.",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument(
        "--estimate-input-cost",
        action="store_true",
        help=(
            "Estimate prompt/input token counts and (if available) prompt cost for the requests "
            "that would be made, without calling the LLM. Respects --cache-only and caches per run seed."
        ),
    )

    return parser.parse_args()


def compute_run_aggregates(metrics: list[dict]) -> dict[str, float]:
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
        if (micro_precision + micro_recall)
        else 0.0
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()

    if not 0.0 <= args.lenience <= 1.0:
        raise ValueError(f"Lenience must be within [0, 1]; received {args.lenience}")
    if args.runs < 1:
        raise ValueError(f"--runs must be >= 1; received {args.runs}")
    if args.seed_step < 0:
        raise ValueError(f"--seed-step must be >= 0; received {args.seed_step}")
    if args.temperature < 0.0:
        raise ValueError(f"--temperature must be >= 0.0; received {args.temperature}")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError(f"--top-p must be within (0, 1]; received {args.top_p}")

    use_full_graph = args.full_graph
    model_name = args.model
    base_url = args.base_url
    depth_value = None if use_full_graph else args.depth
    depth_label = "full" if use_full_graph else str(depth_value)
    lenience = args.lenience
    temperature = args.temperature
    top_p = args.top_p
    base_seed = int(args.seed)
    runs = 1 if args.profile_subgraphs else args.runs
    seed_step = int(args.seed_step)

    output_paths = build_output_paths(
        model_name=model_name,
        depth_label=depth_label,
        lenience=lenience,
        temperature=temperature,
        top_p=top_p,
        base_seed=base_seed,
        runs=runs,
        seed_step=seed_step,
    )

    dataset = load_dataset()
    documentation_queries = sorted(DOCUMENTATION_QUERY_DIR.glob("*.rq"))
    queries = load_queries(documentation_queries)

    normaliser = ValueNormalizer(dataset)
    mode_label = "full" if use_full_graph else "subgraph"

    cache: dict[str, dict] = {} if args.profile_subgraphs else load_cache(output_paths.cache)
    cache_modified = False

    # Do not initialise the client in modes that must not call the LLM.
    api_key = None if args.profile_subgraphs else read_api_key(API_KEY_PATH)
    client = None
    if (not args.profile_subgraphs) and (not args.estimate_input_cost):
        if api_key is None:
            raise RuntimeError("API key missing.")
        kwargs: dict[str, object] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)  # type: ignore[arg-type]

    # Precompute deterministic contexts once (so runs only vary by LLM sampling/seed).
    question_contexts: list[QuestionContext] = []

    if use_full_graph:
        context_text = stable_rdf_text(dataset)
        context_hash = compute_context_hash(context_text)
        monitoring_dict = summarise_subgraph(set(), None, dataset, mode_label).asdict()

        for path, query_text in queries.items():
            question = extract_question_text(path)
            columns, gold_rows = run_sparql(dataset, query_text)
            gold_rows_list, gold_set = canonicalise_rows(columns, gold_rows, normaliser)
            gold_rows_canonical = [list(row) for row in sorted(gold_set)]
            max_mismatches = math.floor(len(columns) * lenience)

            question_contexts.append(
                QuestionContext(
                    path=path,
                    question=question,
                    query_text=query_text,
                    columns=columns,
                    gold_rows_list=gold_rows_list,
                    gold_rows_canonical=gold_rows_canonical,
                    max_mismatches=max_mismatches,
                    context_text=context_text,
                    context_hash=context_hash,
                    monitoring_dict=monitoring_dict,
                )
            )
    else:
        for path, query_text in queries.items():
            question = extract_question_text(path)
            columns, gold_rows = run_sparql(dataset, query_text)
            gold_rows_list, gold_set = canonicalise_rows(columns, gold_rows, normaliser)
            gold_rows_canonical = [list(row) for row in sorted(gold_set)]
            max_mismatches = math.floor(len(columns) * lenience)

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
            monitoring_dict = summarise_subgraph(
                seeds,
                depth_value or DEFAULT_SUBGRAPH_DEPTH,
                subgraph,
                mode_label,
            ).asdict()

            LOGGER.info(
                "Subgraph retrieval for %s retrieved %d triples (depth=%s, seeds=%d)",
                path.name,
                monitoring_dict.get("triple_count", 0),
                depth_label,
                len(monitoring_dict.get("seeds", [])),
            )

            context_text = stable_rdf_text(subgraph)
            context_hash = compute_context_hash(context_text)

            question_contexts.append(
                QuestionContext(
                    path=path,
                    question=question,
                    query_text=query_text,
                    columns=columns,
                    gold_rows_list=gold_rows_list,
                    gold_rows_canonical=gold_rows_canonical,
                    max_mismatches=max_mismatches,
                    context_text=context_text,
                    context_hash=context_hash,
                    monitoring_dict=monitoring_dict,
                )
            )

    # Profiling mode: write contexts and monitoring, no LLM calls.
    if args.profile_subgraphs:
        profile_entries: list[dict[str, object]] = []
        output_paths.profile_dir.mkdir(parents=True, exist_ok=True)

        for qc in question_contexts:
            profile_path = output_paths.profile_dir / f"{qc.path.stem}.nt"
            profile_path.write_text(qc.context_text, encoding="utf-8")
            if qc.monitoring_dict.get("triple_count", 0) == 0:
                LOGGER.warning(
                    "Profiling: %s produced 0 triples. Seeds=%s",
                    qc.path.name,
                    ", ".join(qc.monitoring_dict.get("seeds", [])) or "(none)",
                )
            profile_entries.append(
                {
                    "question": qc.question,
                    "query_path": str(qc.path),
                    "mode": mode_label,
                    "depth": depth_label,
                    "model": model_name,
                    "lenience": lenience,
                    "max_mismatches": qc.max_mismatches,
                    "subgraph_monitoring": qc.monitoring_dict,
                    "subgraph_path": str(profile_path),
                    "context_hash": qc.context_hash,
                }
            )

        output_paths.profile_summary.parent.mkdir(parents=True, exist_ok=True)
        profile_summary = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "model": model_name,
            "mode": mode_label,
            "subgraph_depth": depth_label,
            "lenience": lenience,
            "temperature": temperature,
            "top_p": top_p,
            "seed": base_seed,
            "runs": runs,
            "seed_step": seed_step,
            "profiles": profile_entries,
        }
        output_paths.profile_summary.write_text(
            json.dumps(profile_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for entry in profile_entries:
            monitoring = entry["subgraph_monitoring"]
            print(
                f"{Path(entry['query_path']).name}: triples={monitoring['triple_count']} "
                f"seeds={len(monitoring['seeds'])} depth={depth_label}"
            )
        print(f"\nProfiling summaries written to {output_paths.profile_summary}")
        return

    # Cost estimation mode: no LLM calls; can optionally use OpenRouter pricing if base_url is OpenRouter.
    if args.estimate_input_cost:
        if api_key is None:
            raise RuntimeError("API key missing.")

        pricing: dict[str, str] | None = None
        pricing_fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        if is_openrouter_base_url(base_url):
            try:
                pricing = get_openrouter_prompt_pricing(api_key, model_name)
                LOGGER.info(f"Got pricing {pricing}")
            except Exception as exc:
                LOGGER.warning("Failed to fetch OpenRouter pricing for %s: %s", model_name, exc)
                pricing = None

        estimate_runs: list[dict[str, object]] = []
        grand_calls = 0
        grand_tokens = 0
        grand_cost = Decimal("0")
        has_cost = pricing is not None

        for run_idx in range(runs):
            run_seed = base_seed + run_idx * seed_step
            run_calls = 0
            run_tokens = 0
            run_cost = Decimal("0")
            questions_out: list[dict[str, object]] = []

            for qc in question_contexts:
                cache_key = make_cache_key(
                    qc.path,
                    qc.question,
                    qc.query_text,
                    qc.context_hash,
                    qc.columns,
                    depth_label,
                    model_name,
                    temperature=temperature,
                    top_p=top_p,
                    seed=run_seed,
                    response_format="json_object",
                )
                cached = get_cached_payload(cache, cache_key)
                would_call = (cached is None) and (not args.cache_only)

                messages = build_messages(qc.question, qc.context_text, qc.columns)
                est = estimate_input_cost(
                    messages=messages,
                    model_name=model_name,
                    base_url=base_url,
                    pricing=pricing,
                )

                prompt_tokens = int(est["prompt_tokens_est"])  # type: ignore[arg-type]
                run_tokens += prompt_tokens if would_call else 0
                if would_call:
                    run_calls += 1
                    if est.get("input_cost_est_usd") is not None:
                        run_cost += _decimal_or_zero(est["input_cost_est_usd"])

                questions_out.append(
                    {
                        "question": qc.question,
                        "query_path": str(qc.path),
                        "context_hash": qc.context_hash,
                        "subgraph_triples": int(qc.monitoring_dict.get("triple_count", 0)),
                        "cached": cached is not None,
                        "would_call": would_call,
                        **est,
                    }
                )

            grand_calls += run_calls
            grand_tokens += run_tokens
            grand_cost += run_cost

            estimate_runs.append(
                {
                    "run_index": run_idx,
                    "seed": run_seed,
                    "would_call_count": run_calls,
                    "prompt_tokens_total_est": run_tokens,
                    "input_cost_total_est_usd": format(run_cost, "f") if has_cost else None,
                    "questions": questions_out,
                }
            )

            print(f"\nCost estimate run {run_idx + 1}/{runs} (seed={run_seed})")
            print(f"- Would call LLM: {run_calls} requests")
            print(f"- Prompt tokens (est): {run_tokens}")
            if has_cost:
                print(f"- Prompt cost (est): ${format(run_cost, 'f')}")
            else:
                print("- Prompt cost (est): unavailable (no pricing source)")

        payload = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "pricing_fetched_at": pricing_fetched_at,
            "mode": mode_label,
            "subgraph_depth": depth_label,
            "model": model_name,
            "base_url": base_url,
            "response_format": "json_object",
            "temperature": temperature,
            "top_p": top_p,
            "base_seed": base_seed,
            "runs": runs,
            "seed_step": seed_step,
            "cache_only": bool(args.cache_only),
            "pricing": pricing,
            "grand_totals": {
                "would_call_count": grand_calls,
                "prompt_tokens_total_est": grand_tokens,
                "input_cost_total_est_usd": format(grand_cost, "f") if has_cost else None,
            },
            "runs_detail": estimate_runs,
        }

        output_paths.cost_estimate.parent.mkdir(parents=True, exist_ok=True)
        output_paths.cost_estimate.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        print("\nAcross runs (estimates for calls that would be made)")
        print(f"- Would call LLM: {grand_calls} requests")
        print(f"- Prompt tokens (est): {grand_tokens}")
        if has_cost:
            print(f"- Prompt cost (est): ${format(grand_cost, 'f')}")
        else:
            print("- Prompt cost (est): unavailable (no pricing source)")
        print(f"\nWrote {output_paths.cost_estimate}")
        return

    # Execute multiple runs and aggregate.
    per_run_results: list[dict[str, object]] = []
    skipped_due_to_cache_by_run: list[list[str]] = []
    per_question_metrics_by_run: list[list[dict]] = []

    for run_idx in range(runs):
        run_seed = base_seed + run_idx * seed_step
        LOGGER.info("=== Run %d/%d (seed=%d) ===", run_idx + 1, runs, run_seed)

        metrics: list[dict] = []
        skipped_due_to_cache: list[str] = []
        per_question_metrics: list[dict] = []

        for qc in question_contexts:
            cache_key = make_cache_key(
                qc.path,
                qc.question,
                qc.query_text,
                qc.context_hash,
                qc.columns,
                depth_label,
                model_name,
                temperature=temperature,
                top_p=top_p,
                seed=run_seed,
                response_format="json_object",
            )

            cached = get_cached_payload(cache, cache_key)

            if cached is None and args.cache_only:
                fallback = fallback_cache_entry(
                    cache,
                    qc.path,
                    qc.columns,
                    len(qc.gold_rows_canonical),
                    LEGACY_CACHE_PATHS,
                )
                if fallback is not None:
                    cache[cache_key] = {
                        "columns": fallback.get("columns", []),
                        "rows": fallback.get("rows", []),
                        "_meta": {
                            "path": str(qc.path),
                            "question": qc.question,
                            "columns": list(qc.columns),
                            "context_hash": qc.context_hash,
                            "depth": depth_label,
                            "model": model_name,
                            "mode": mode_label,
                            "lenience": lenience,
                            "max_mismatches": qc.max_mismatches,
                            "subgraph_monitoring": qc.monitoring_dict,
                            "temperature": temperature,
                            "top_p": top_p,
                            "seed": run_seed,
                            "response_format": "json_object",
                            "fallback_used": True,
                        },
                    }
                    cached = get_cached_payload(cache, cache_key)
                    cache_modified = True

            if cached is None:
                if args.cache_only:
                    skipped_due_to_cache.append(qc.question)
                    continue
                if client is None:
                    raise RuntimeError("LLM client not initialised.")

                model_json, llm_meta = call_llm(
                    client,
                    model_name,
                    qc.question,
                    qc.context_text,
                    qc.columns,
                    temperature=temperature,
                    top_p=top_p,
                    seed=run_seed,
                )
                cache[cache_key] = {
                    "columns": model_json.get("columns", []),
                    "rows": model_json.get("rows", []),
                    "_meta": {
                        "path": str(qc.path),
                        "question": qc.question,
                        "columns": list(qc.columns),
                        "context_hash": qc.context_hash,
                        "depth": depth_label,
                        "model": model_name,
                        "mode": mode_label,
                        "lenience": lenience,
                        "max_mismatches": qc.max_mismatches,
                        "subgraph_monitoring": qc.monitoring_dict,
                        **llm_meta,
                    },
                }
                cached = get_cached_payload(cache, cache_key)
                cache_modified = True

            assert cached is not None
            model_columns = [str(col) for col in cached.get("columns", [])]
            model_rows_raw = cached.get("rows", [])
            meta = cached.get("_meta", {})

            if not isinstance(model_rows_raw, list):
                raise ValueError(f"Model rows must be a list: {cached}")

            model_rows = [
                [str(cell) for cell in row] if isinstance(row, (list, tuple)) else [str(row)]
                for row in model_rows_raw
            ]

            pred_rows_list: list[tuple[str, ...]] = []
            canonical_pred_rows: list[list[str]] = []
            if model_columns and len(model_columns) == len(qc.columns):
                pred_rows_list, pred_set = canonicalise_rows(qc.columns, model_rows, normaliser)
                canonical_pred_rows = [list(row) for row in sorted(pred_set)]

            precision, recall, f1, tp, fp, fn = score_predictions(
                qc.gold_rows_list,
                pred_rows_list,
                qc.max_mismatches,
            )

            entry = {
                "path": qc.path,
                "question": qc.question,
                "columns": qc.columns,
                "gold_rows": qc.gold_rows_canonical,
                "pred_rows": canonical_pred_rows,
                "pred_rows_raw": model_rows,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "context_hash": qc.context_hash,
                "max_mismatches": qc.max_mismatches,
                "subgraph_monitoring": qc.monitoring_dict,
                "llm_meta": {
                    "system_fingerprint": meta.get("system_fingerprint"),
                    "seed": meta.get("seed", run_seed),
                    "temperature": meta.get("temperature", temperature),
                    "top_p": meta.get("top_p", top_p),
                    "model": meta.get("model", model_name),
                },
            }
            metrics.append(entry)
            per_question_metrics.append(
                {
                    "question": qc.question,
                    "query_path": str(qc.path),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "system_fingerprint": meta.get("system_fingerprint"),
                }
            )

        run_aggs = compute_run_aggregates(metrics)
        per_run_results.append(
            {
                "run_index": run_idx,
                "seed": run_seed,
                "skipped_questions": skipped_due_to_cache,
                "aggregate": run_aggs,
            }
        )
        skipped_due_to_cache_by_run.append(skipped_due_to_cache)
        per_question_metrics_by_run.append(metrics)

        print(f"\nRun {run_idx + 1}/{runs} (seed={run_seed})")
        print("Micro:", f"P={run_aggs['micro_precision']:.4f}",
              f"R={run_aggs['micro_recall']:.4f}",
              f"F1={run_aggs['micro_f1']:.4f}")
        print("Macro:", f"P={run_aggs['macro_precision']:.4f}",
              f"R={run_aggs['macro_recall']:.4f}",
              f"F1={run_aggs['macro_f1']:.4f}")

    if cache_modified:
        save_cache(output_paths.cache, cache)
        LOGGER.info("Updated cache written to %s", output_paths.cache)

    # Aggregate across runs (mean/stddev).
    micro_p = [r["aggregate"]["micro_precision"] for r in per_run_results]  # type: ignore[index]
    micro_r = [r["aggregate"]["micro_recall"] for r in per_run_results]     # type: ignore[index]
    micro_f1 = [r["aggregate"]["micro_f1"] for r in per_run_results]        # type: ignore[index]
    macro_p = [r["aggregate"]["macro_precision"] for r in per_run_results]  # type: ignore[index]
    macro_r = [r["aggregate"]["macro_recall"] for r in per_run_results]     # type: ignore[index]
    macro_f1 = [r["aggregate"]["macro_f1"] for r in per_run_results]        # type: ignore[index]

    agg_summary = {
        "micro_precision": {"mean": mean_std(micro_p)[0], "stddev": mean_std(micro_p)[1]},
        "micro_recall": {"mean": mean_std(micro_r)[0], "stddev": mean_std(micro_r)[1]},
        "micro_f1": {"mean": mean_std(micro_f1)[0], "stddev": mean_std(micro_f1)[1]},
        "macro_precision": {"mean": mean_std(macro_p)[0], "stddev": mean_std(macro_p)[1]},
        "macro_recall": {"mean": mean_std(macro_r)[0], "stddev": mean_std(macro_r)[1]},
        "macro_f1": {"mean": mean_std(macro_f1)[0], "stddev": mean_std(macro_f1)[1]},
    }

    print("\nAcross runs")
    print("-----------")
    print(f"Micro Precision: mean={agg_summary['micro_precision']['mean']:.4f} "
          f"std={agg_summary['micro_precision']['stddev']:.4f}")
    print(f"Micro Recall:    mean={agg_summary['micro_recall']['mean']:.4f} "
          f"std={agg_summary['micro_recall']['stddev']:.4f}")
    print(f"Micro F1:        mean={agg_summary['micro_f1']['mean']:.4f} "
          f"std={agg_summary['micro_f1']['stddev']:.4f}")
    print(f"Macro Precision: mean={agg_summary['macro_precision']['mean']:.4f} "
          f"std={agg_summary['macro_precision']['stddev']:.4f}")
    print(f"Macro Recall:    mean={agg_summary['macro_recall']['mean']:.4f} "
          f"std={agg_summary['macro_recall']['stddev']:.4f}")
    print(f"Macro F1:        mean={agg_summary['macro_f1']['mean']:.4f} "
          f"std={agg_summary['macro_f1']['stddev']:.4f}")

    # Per-question mean/std across runs (precision/recall/f1).
    per_question_accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
    per_question_paths: dict[str, str] = {}
    for run_metrics in per_question_metrics_by_run:
        for entry in run_metrics:
            q = entry["question"]
            per_question_paths[q] = str(entry["path"])
            per_question_accum[q]["precision"].append(entry["precision"])
            per_question_accum[q]["recall"].append(entry["recall"])
            per_question_accum[q]["f1"].append(entry["f1"])

    per_question_summary = []
    for q, vals in sorted(per_question_accum.items(), key=lambda kv: kv[0]):
        p_mean, p_std = mean_std(vals["precision"])
        r_mean, r_std = mean_std(vals["recall"])
        f_mean, f_std = mean_std(vals["f1"])
        per_question_summary.append(
            {
                "question": q,
                "query_path": per_question_paths.get(q, ""),
                "precision": {"mean": p_mean, "stddev": p_std},
                "recall": {"mean": r_mean, "stddev": r_std},
                "f1": {"mean": f_mean, "stddev": f_std},
            }
        )

    # Markdown report.
    report_lines = [
        "# Competency LLM Evaluation",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Mode: {mode_label}",
        f"- Subgraph depth: {depth_label}",
        f"- Model: {model_name}",
        f"- Lenience: {lenience:.2f}",
        f"- Temperature: {temperature:.3f}",
        f"- Top-p: {top_p:.3f}",
        f"- Base seed: {base_seed}",
        f"- Runs: {runs} (seed_step={seed_step})",
        "",
        "## Across-Run Aggregates (Mean ± Stddev)",
        "",
        f"- Micro Precision: {agg_summary['micro_precision']['mean']:.4f} ± {agg_summary['micro_precision']['stddev']:.4f}",
        f"- Micro Recall: {agg_summary['micro_recall']['mean']:.4f} ± {agg_summary['micro_recall']['stddev']:.4f}",
        f"- Micro F1: {agg_summary['micro_f1']['mean']:.4f} ± {agg_summary['micro_f1']['stddev']:.4f}",
        f"- Macro Precision: {agg_summary['macro_precision']['mean']:.4f} ± {agg_summary['macro_precision']['stddev']:.4f}",
        f"- Macro Recall: {agg_summary['macro_recall']['mean']:.4f} ± {agg_summary['macro_recall']['stddev']:.4f}",
        f"- Macro F1: {agg_summary['macro_f1']['mean']:.4f} ± {agg_summary['macro_f1']['stddev']:.4f}",
        "",
        "## Per-Run Metrics",
        "",
        "| Run | Seed | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Skipped |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in per_run_results:
        ag = r["aggregate"]  # type: ignore[index]
        skipped = len(r["skipped_questions"])  # type: ignore[index]
        report_lines.append(
            f"| {r['run_index']} | {r['seed']} | "
            f"{ag['micro_precision']:.4f} | {ag['micro_recall']:.4f} | {ag['micro_f1']:.4f} | "
            f"{ag['macro_precision']:.4f} | {ag['macro_recall']:.4f} | {ag['macro_f1']:.4f} | {skipped} |"
        )

    report_lines.extend(
        [
            "",
            "## Per-Question Metrics (Mean ± Stddev Across Runs)",
            "",
            "| Question | Mean P | Std P | Mean R | Std R | Mean F1 | Std F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_question_summary:
        report_lines.append(
            f"| {row['question']} | "
            f"{row['precision']['mean']:.4f} | {row['precision']['stddev']:.4f} | "
            f"{row['recall']['mean']:.4f} | {row['recall']['stddev']:.4f} | "
            f"{row['f1']['mean']:.4f} | {row['f1']['stddev']:.4f} |"
        )

    output_paths.report.parent.mkdir(parents=True, exist_ok=True)
    output_paths.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # JSON summary.
    summary_payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "mode": mode_label,
        "model": model_name,
        "subgraph_depth": depth_label,
        "lenience": lenience,
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": base_seed,
        "runs": runs,
        "seed_step": seed_step,
        "across_runs": agg_summary,
        "per_run": per_run_results,
        "per_question_across_runs": per_question_summary,
    }
    output_paths.summary.parent.mkdir(parents=True, exist_ok=True)
    output_paths.summary.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
