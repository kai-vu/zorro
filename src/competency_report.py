"""Generate a markdown report for competency question queries.

This module evaluates each SPARQL query stored under the ``queries`` directory
against the RDF datasets in ``generated-rdf`` and writes a concise report
summarizing execution status and result counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from rdflib import ConjunctiveGraph
from rdflib.query import Result


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryEvaluationConfig:
    """Runtime configuration for generating the competency question report.

    Attributes:
        rdf_dir: Directory containing RDF files (TTL or TriG) to load.
        query_dir: Directory containing SPARQL query files with ``.rq`` suffix.
        report_path: Output path for the generated markdown report.
        log_level: Logging level to configure for the run.
    """

    rdf_dir: Path
    query_dir: Path
    report_path: Path
    log_level: int


@dataclass(frozen=True)
class QueryTestResult:
    """Outcome of evaluating a single SPARQL query."""

    query_path: Path
    status: str
    row_count: int | None
    duration_seconds: float
    note: str | None = None
    columns: list[str] | None = None
    row_values: list[tuple[str, ...]] | None = None

    @property
    def display_name(self) -> str:
        """Return the filename without directory segments."""

        return self.query_path.name


def parse_args(argv: Sequence[str] | None = None) -> QueryEvaluationConfig:
    """Parse CLI arguments into a ``QueryEvaluationConfig`` instance.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        A populated configuration dataclass.
    """

    parser = argparse.ArgumentParser(
        description="Generate a markdown report for competency question queries."
    )
    parser.add_argument(
        "--rdf-dir",
        type=Path,
        default=Path("generated-rdf"),
        help="Directory containing TTL/TriG files (default: generated-rdf).",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=Path("queries"),
        help="Directory containing .rq query files (default: queries).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("queries/reports/competency-report.md"),
        help=(
            "Path where the markdown report will be written "
            "(default: queries/reports/competency-report.md)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    args = parser.parse_args(argv)
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    return QueryEvaluationConfig(
        rdf_dir=args.rdf_dir,
        query_dir=args.query_dir,
        report_path=args.report_path,
        log_level=log_level,
    )


def load_graph(rdf_dir: Path) -> ConjunctiveGraph:
    """Load all TTL and TriG files from ``rdf_dir`` into a single graph.

    Args:
        rdf_dir: Directory that contains RDF serializations to parse.

    Returns:
        A populated ``ConjunctiveGraph`` instance.

    Raises:
        FileNotFoundError: If the RDF directory does not exist.
    """

    if not rdf_dir.exists():
        raise FileNotFoundError(f"RDF directory does not exist: {rdf_dir}")

    graph = ConjunctiveGraph()
    for rdf_path in sorted(rdf_dir.glob("*")):
        suffix = rdf_path.suffix.lower()
        if suffix not in {".ttl", ".trig"}:
            continue
        format_hint = "turtle" if suffix == ".ttl" else "trig"
        try:
            LOGGER.debug("Loading RDF file: %s", rdf_path)
            graph.parse(str(rdf_path), format=format_hint)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to parse %s: %s", rdf_path, exc)
    LOGGER.info("Loaded %s triples from %s", len(graph), rdf_dir)
    return graph


def execute_query(graph: ConjunctiveGraph, query_path: Path) -> QueryTestResult:
    """Execute a single SPARQL query and capture the outcome.

    Args:
        graph: RDF graph to query.
        query_path: Path to the ``.rq`` file.

    Returns:
        A ``QueryTestResult`` summarizing execution status.
    """

    query_text = query_path.read_text(encoding="utf-8")
    start = time.perf_counter()
    try:
        LOGGER.debug("Executing query %s", query_path)
        result: Result = graph.query(query_text)
        # Materialize results to ensure we capture row counts now.
        rows = list(result)
        columns = [str(var) for var in result.vars]
        row_values = [
            tuple("" if value is None else str(value) for value in record)
            for record in rows
        ]
        duration = time.perf_counter() - start
        status = "pass" if rows else "empty"
        note = None
        if not rows:
            note = "No rows returned."
        return QueryTestResult(
            query_path=query_path,
            status=status,
            row_count=len(row_values),
            duration_seconds=duration,
            note=note,
            columns=columns,
            row_values=row_values,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - start
        LOGGER.warning("Query %s failed: %s", query_path, exc)
        note = textwrap.shorten(str(exc), width=120, placeholder="…")
        return QueryTestResult(
            query_path=query_path,
            status="error",
            row_count=None,
            duration_seconds=duration,
            note=note,
            columns=None,
            row_values=None,
        )


def evaluate_queries(graph: ConjunctiveGraph, query_dir: Path) -> list[QueryTestResult]:
    """Evaluate all query files in ``query_dir``.

    Args:
        graph: RDF graph to use for evaluation.
        query_dir: Directory containing ``.rq`` files.

    Returns:
        List of ``QueryTestResult`` instances sorted by filename.
    """

    if not query_dir.exists():
        raise FileNotFoundError(f"Query directory does not exist: {query_dir}")

    results: list[QueryTestResult] = []
    for query_path in sorted(query_dir.rglob("*.rq")):
        if query_dir.parent.name.startswith('.'):
            continue
        results.append(execute_query(graph, query_path))
    return results


def render_report(results: Iterable[QueryTestResult], output_path: Path) -> None:
    """Write the markdown report to ``output_path``.

    Args:
        results: Iterable of query execution results.
        output_path: Destination path for the markdown file.
    """

    results = list(results)

    def extract_question(result: QueryTestResult) -> str:
        for line in result.query_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return result.query_path.stem

    grouped: dict[str, list[QueryTestResult]] = {}
    for result in results:
        relative = result.query_path.relative_to(Path("queries"))
        category = relative.parts[0] if len(relative.parts) > 1 else "."
        grouped.setdefault(category, []).append(result)

    passed = sum(1 for result in results if result.status == "pass")
    empty = sum(1 for result in results if result.status == "empty")
    errored = sum(1 for result in results if result.status == "error")
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    header_lines = [
        "# Competency Question Report",
        "",
        f"- Generated: {timestamp}",
        f"- Total queries: {len(results)}",
        f"- Passed: {passed}",
        f"- Empty: {empty}",
        f"- Errors: {errored}",
    ]

    table_lines: list[str] = []
    for category in sorted(grouped):
        table_lines.append(f"\n## {category.capitalize()} Queries\n")
        table_lines.append("| Question | Status | Rows | Duration (s) | Notes |")
        table_lines.append("| --- | --- | ---: | ---: | --- |")
        for result in sorted(grouped[category], key=lambda r: r.query_path.name):
            question = extract_question(result)
            row_count_str = "" if result.row_count is None else str(result.row_count)
            duration_str = f"{result.duration_seconds:.3f}"
            note = result.note or ""
            table_lines.append(
                f"| {question} | {result.status} | {row_count_str} | {duration_str} | {note} |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header_lines + table_lines) + "\n", encoding="utf-8")

    cache_payload = {
        "generated": timestamp,
        "total": len(results),
        "passed": passed,
        "empty": empty,
        "errors": errored,
        "results": {
            category: [
                {
                    "question": extract_question(result),
                    "query_path": str(result.query_path),
                    "status": result.status,
                    "row_count": result.row_count,
                    "duration_seconds": result.duration_seconds,
                    "note": result.note,
                    "columns": result.columns or [],
                    "rows": [list(row) for row in (result.row_values or [])],
                }
                for result in sorted(grouped[category], key=lambda r: r.query_path.name)
            ]
            for category in sorted(grouped)
        },
    }
    cache_path = output_path.with_name("competency_sparql_cache.json")
    cache_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote report to %s", output_path)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    config = parse_args(argv)
    logging.basicConfig(
        level=config.log_level,
        format="%(levelname)s %(name)s - %(message)s",
    )
    graph = load_graph(config.rdf_dir)
    results = evaluate_queries(graph, config.query_dir)
    render_report(results, config.report_path)


if __name__ == "__main__":
    main()
