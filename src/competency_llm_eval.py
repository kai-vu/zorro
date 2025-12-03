"""Evaluate LLM answers to documentation-based competency questions.

The script loads only the documentation-backed competency queries, runs their
SPARQL definitions to obtain gold-standard answers, and prompts an LLM with the
entire non-extraction RDF graph as textual context. Responses are cached on
disk to avoid repeated API calls, and basic precision/recall/F1 metrics are
reported.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI
from rdflib import Dataset


MODEL = "gpt-5-nano"
API_KEY_PATH = Path("openai-key.txt")
QUERY_DIR = Path("queries")
DOCUMENTATION_QUERY_DIR = QUERY_DIR / "documentation"
GENERATED_RDF_DIR = Path("generated-rdf")
EXCLUDED_PREFIXES = ("extractions_",)
CACHE_PATH = Path("queries/reports/competency_llm_cache.json")

LOGGER = logging.getLogger(__name__)


def load_graph() -> Dataset:
    dataset = Dataset()
    for path in sorted(GENERATED_RDF_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix not in {".ttl", ".trig"}:
            continue
        if path.name.startswith(EXCLUDED_PREFIXES):
            continue
        fmt = "turtle" if suffix == ".ttl" else "trig"
        dataset.parse(path, format=fmt)
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
        "- Use empty strings for unknown values. Return an empty array for \"rows\" if the context does not justify an answer.\n"
        "- Do not include explanations or additional keys."
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


def to_tuple_set(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> set[tuple[str, ...]]:
    return {tuple(str(value) for value in row[: len(columns)]) for row in rows}


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


def evaluate(
    gold: set[tuple[str, ...]],
    pred: set[tuple[str, ...]],
) -> tuple[float, float, float, int, int, int]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if not pred:
        fn = len(gold)
        return 0.0, 1.0 if gold else 1.0, 0.0, 0, 0, fn
    if not gold:
        fp = len(pred)
        return 0.0, 0.0, 0.0, 0, fp, 0
    intersection = gold & pred
    precision = len(intersection) / len(pred)
    recall = len(intersection) / len(gold)
    if precision + recall == 0:
        return precision, recall, 0.0, len(intersection), len(pred) - len(intersection), len(gold) - len(intersection)
    f1 = 2 * precision * recall / (precision + recall)
    return (
        precision,
        recall,
        f1,
        len(intersection),
        len(pred) - len(intersection),
        len(gold) - len(intersection),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    graph = load_graph()
    documentation_queries = sorted(DOCUMENTATION_QUERY_DIR.glob("*.rq"))
    queries = load_queries(documentation_queries)

    serialized = graph.serialize(format="trig")
    if isinstance(serialized, bytes):
        serialized = serialized.decode("utf-8")

    client = OpenAI(api_key=API_KEY_PATH.read_text(encoding="utf-8").strip())
    cache = load_cache()
    context_hash = compute_context_hash(serialized)
    cache_modified = False

    metrics: list[dict] = []
    summary_lines: list[str] = []

    for path, query_text in queries.items():
        question = extract_question_text(path)
        columns, gold_rows = run_sparql(graph, query_text)
        gold_set = to_tuple_set(columns, gold_rows)

        cache_key = make_cache_key(path, question, query_text, context_hash, columns)
        if cache_key in cache:
            LOGGER.info("Using cached response for %s (%s)", path.name, question)
            model_json = cache[cache_key]
        else:
            LOGGER.info("Querying LLM for %s (%s)", path.name, question)
            model_json = call_llm(client, question, serialized, columns)
            cache[cache_key] = model_json
            cache_modified = True

        model_columns = [str(col) for col in model_json.get("columns", [])]
        model_rows = model_json.get("rows", [])
        if not isinstance(model_rows, list):
            raise ValueError(f"Model rows must be a list: {model_json}")

        pred_set: set[tuple[str, ...]] = set()
        if model_columns and len(model_columns) == len(columns):
            pred_set = to_tuple_set(columns, model_rows)

        precision, recall, f1, tp, fp, fn = evaluate(gold_set, pred_set)
        metrics.append(
            {
                "path": path,
                "question": question,
                "columns": columns,
                "gold_rows": sorted(list(gold_set)),
                "pred_rows": sorted(list(pred_set)),
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
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else math.nan
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else math.nan
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else math.nan
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
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "results": [
            {
                "question": entry["question"],
                "query_path": str(entry["path"]),
                "columns": entry["columns"],
                "gold_rows": entry["gold_rows"],
                "pred_rows": entry["pred_rows"],
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
