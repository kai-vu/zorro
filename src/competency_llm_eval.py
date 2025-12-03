"""Evaluate LLM answers to documentation-based competency questions.

The script loads only the documentation-backed competency queries, runs their
SPARQL definitions to obtain gold-standard answers, and prompts an LLM with the
entire non-extraction RDF graph as textual context. Responses are cached on
disk to avoid repeated API calls, and basic precision/recall/F1 metrics are
reported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import statistics
import re
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI
from rdflib import Dataset, URIRef
from rdflib.namespace import RDFS


MODEL = "gpt-5-nano"
API_KEY_PATH = Path("openai-key.txt")
QUERY_DIR = Path("queries")
DOCUMENTATION_QUERY_DIR = QUERY_DIR / "documentation"
GENERATED_RDF_DIR = Path("generated-rdf")
EXCLUDED_PREFIXES = ("extractions_",)
CACHE_PATH = Path("queries/reports/competency_llm_cache.json")
SUBGRAPH_CACHE_PATH = Path("queries/reports/competency_llm_subgraph_cache.json")

LOGGER = logging.getLogger(__name__)


class ValueNormalizer:
    """Convert predicted strings into canonical identifiers (URIs when possible)."""

    def __init__(self, dataset: Dataset):
        graph = dataset.graph()
        self.ns_manager = graph.namespace_manager

        label_candidates = {}
        local_candidates = {}

        for subject, _, label in graph.triples((None, RDFS.label, None)):
            if not isinstance(subject, URIRef):
                continue
            label_text = str(label)
            key = self._normalise_key(label_text)
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


def load_graph() -> Dataset:
    dataset = Dataset()
    for path in sorted(GENERATED_RDF_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix not in {".ttl", ".trig"}:
            continue
        if path.name.startswith(EXCLUDED_PREFIXES):
            continue
        fmt = "turtle" if suffix == ".ttl" else "trig"
        dataset.parse(str(path), format=fmt)
    dataset.default_union = True
    return dataset


def load_queries(paths: Iterable[Path]) -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in paths}


def extract_question_text(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    raise ValueError(f"No question comment found in {path}")


def run_sparql(graph: Dataset, query: str) -> tuple[list[str], list[tuple[str, ...]]]:
    result = graph.query(query)
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


def call_llm(
    client: OpenAI,
    question: str,
    context_text: str,
    expected_columns: Sequence[str],
) -> dict:
    user_prompt = (
        "Context (Turtle serialization of the ontology and documentation-derived data):\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        f"- Return ONLY JSON with two keys: \"columns\" and \"rows\".\n"
        f"- The \"columns\" array MUST exactly match this list and order: {list(expected_columns)}.\n"
        "- The \"rows\" array must contain one array per answer, aligned to the column order.\n"
        "- Every resource MUST be written as the exact IRI found in the context (full IRI preferred, CURIE allowed only if the prefix is defined).\n"
        "- Use empty strings for unknown values. Return an empty array for \"rows\" if the context does not justify an answer.\n"
        "- Do not include explanations or additional keys.\n"
        'Example: {"columns": ["https://example.org/a", "https://example.org/b"], "rows": [["https://example.org/x", "https://example.org/y"]]}'
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer maintenance knowledge graph questions using only the provided RDF context. "
                "Do not rely on outside knowledge."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(strip_code_fences(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON: {content}") from exc


def canonicalise_rows(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    normaliser: ValueNormalizer,
) -> tuple[list[tuple[str, ...]], set[tuple[str, ...]]]:
    canonical_rows: list[tuple[str, ...]] = []
    for row in rows:
        canonical_rows.append(tuple(normaliser.canonical(str(value)) for value in row[: len(columns)]))
    return canonical_rows, set(canonical_rows)


def compute_context_hash(context_text: str) -> str:
    return hashlib.sha256(context_text.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Cache at %s is invalid JSON; starting fresh", CACHE_PATH)
    return {}


def save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def make_cache_key(
    path: Path,
    question: str,
    query_text: str,
    context_hash: str,
    expected_columns: Sequence[str],
) -> str:
    payload = {
        "path": str(path),
        "question": question,
        "query": query_text,
        "context_hash": context_hash,
        "columns": list(expected_columns),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fetch_cache_entry(cache: dict[str, dict], cache_key: str) -> dict | None:
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
) -> dict | None:
    def _scan(entries: dict[str, dict]) -> dict | None:
        target_columns = list(expected_columns)
        candidates: list[dict] = []
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            meta = entry.get("_meta")
            if isinstance(meta, dict):
                if meta.get("path") == str(path):
                    return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}
                continue
            if entry.get("columns") == target_columns:
                candidates.append(entry)
        if not candidates:
            return None
        if len(candidates) == 1:
            entry = candidates[0]
            return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}
        for entry in candidates:
            if len(entry.get("rows", [])) == gold_size:
                return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}
        entry = candidates[0]
        return {"columns": entry.get("columns", []), "rows": entry.get("rows", [])}

    hit = _scan(cache)
    if hit:
        return hit
    if SUBGRAPH_CACHE_PATH.exists():
        try:
            sub_cache = json.load(SUBGRAPH_CACHE_PATH.open())
        except json.JSONDecodeError:
            sub_cache = {}
        hit = _scan(sub_cache)
        if hit:
            return hit
    return None
    return None


def evaluate(
    gold: set[tuple[str, ...]],
    pred: set[tuple[str, ...]],
) -> tuple[float, float, float, int, int, int]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0, 0, 0, 0

    tp = len(gold & pred)
    fp = len(pred) - tp
    fn = len(gold) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate documentation competency questions with cached LLM answers."
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Reuse cached LLM responses and skip API calls. Queries missing from the cache will be skipped.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    dataset = load_graph()
    documentation_queries = sorted(DOCUMENTATION_QUERY_DIR.glob("*.rq"))
    queries = load_queries(documentation_queries)

    serialized = dataset.serialize(format="trig")
    if isinstance(serialized, bytes):
        serialized = serialized.decode("utf-8")

    client = OpenAI(api_key=API_KEY_PATH.read_text(encoding="utf-8").strip())
    normaliser = ValueNormalizer(dataset)
    cache = load_cache()
    context_hash = compute_context_hash(serialized)
    cache_modified = False

    metrics: list[dict] = []
    summary_lines: list[str] = []
    skipped_due_to_cache: list[str] = []

    for path, query_text in queries.items():
        question = extract_question_text(path)
        columns, gold_rows = run_sparql(dataset, query_text)
        gold_rows_canonical, gold_set = canonicalise_rows(columns, gold_rows, normaliser)
        gold_rows_canonical = [list(row) for row in sorted(gold_set)]

        cache_key = make_cache_key(path, question, query_text, context_hash, columns)
        model_json = fetch_cache_entry(cache, cache_key)
        if model_json is None and args.cache_only:
            model_json = fallback_cache_entry(cache, path, columns, len(gold_rows_canonical))
            if model_json:
                LOGGER.info(
                    "Using cached response for %s (%s) via fallback",
                    path.name,
                    question,
                )
        if model_json is None:
            if args.cache_only:
                LOGGER.warning(
                    "Cache miss for %s (%s) under --cache-only; skipping question.",
                    path.name,
                    question,
                )
                skipped_due_to_cache.append(question)
                continue
            LOGGER.info("Querying LLM for %s (%s)", path.name, question)
            model_json = call_llm(client, question, serialized, columns)
            cache[cache_key] = {
                "columns": model_json.get("columns", []),
                "rows": model_json.get("rows", []),
                "_meta": {
                    "path": str(path),
                    "question": question,
                    "columns": list(columns),
                    "context_hash": context_hash,
                },
            }
            cache_modified = True

        model_columns = [str(col) for col in model_json.get("columns", [])]
        model_rows_raw = model_json.get("rows", [])
        if not isinstance(model_rows_raw, list):
            raise ValueError(f"Model rows must be a list: {model_json}")
        model_rows = [
            [str(cell) for cell in row] if isinstance(row, (list, tuple)) else [str(row)]
            for row in model_rows_raw
        ]

        canonical_pred_rows, pred_set = ([], set())
        if model_columns and len(model_columns) == len(columns):
            canonical_pred_rows, pred_set = canonicalise_rows(columns, model_rows, normaliser)
            canonical_pred_rows = [list(row) for row in sorted(pred_set)]
        else:
            canonical_pred_rows = []

        precision, recall, f1, tp, fp, fn = evaluate(gold_set, pred_set)
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
            }
        )
        summary_lines.append(
            f"{path.name}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
            f"(gold={len(gold_set)}, pred={len(pred_set)})"
        )

    if cache_modified:
        save_cache(cache)
    LOGGER.info("LLM response cache stored at %s", CACHE_PATH)

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

    print("\nPer-question metrics")
    print("--------------------")
    for line in summary_lines:
        print(line)

    print("\nMacro averages")
    print("--------------")
    print(f"Precision: {macro_precision:.2f}")
    print(f"Recall:    {macro_recall:.2f}")
    print(f"F1:        {macro_f1:.2f}")

    print("\nMicro averages")
    print("--------------")
    print(f"Precision: {micro_precision:.2f}")
    print(f"Recall:    {micro_recall:.2f}")
    print(f"F1:        {micro_f1:.2f}")
    if skipped_due_to_cache:
        LOGGER.info(
            "Skipped %d questions due to cache-only mode: %s",
            len(skipped_due_to_cache),
            ", ".join(skipped_due_to_cache),
        )

    report_lines = [
        "# Competency LLM Evaluation Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Context hash: {context_hash}",
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
    report_path = Path("queries/reports/competency-llm-report.md")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    cache_summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "context_hash": context_hash,
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
            }
            for entry in metrics
        ],
    }
    summary_path = Path("queries/reports/competency_llm_results.json")
    summary_path.write_text(json.dumps(cache_summary, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote LLM evaluation summary to %s", summary_path)


if __name__ == "__main__":
    main()
