"""Run a SPARQL query against TTL data and plot results.

This module provides a small utility that reads a SPARQL query from file,
executes it on a TTL graph, and visualizes the counts in a bar chart. It is
intended as a lightweight example of how to explore the knowledge graph.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Sequence

from rdflib import Graph
from rdflib.query import ResultRow


LOGGER = logging.getLogger(__name__)


def configure_matplotlib_environment(cache_dir: Path = Path(".matplotlib-cache")) -> None:
    """Ensure Matplotlib can write its cache even in read-only home directories."""

    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


configure_matplotlib_environment()

import matplotlib.pyplot as plt  # noqa: E402  (import after environment configuration)


@dataclass(frozen=True)
class SPARQLPlotConfig:
    """Configuration for running a SPARQL query and plotting the results."""

    ttl_path: Path
    query_path: Path
    label_columns: tuple[str, ...] | None
    value_column: str | None
    output_path: Path
    top_n: int
    prettify_labels: bool
    wrap_width: int
    style: str | None
    min_value: float
    orientation: str
    title: str
    dpi: int


def parse_args(argv: Sequence[str] | None = None) -> SPARQLPlotConfig:
    """Parse command-line arguments into a configuration object.

    Args:
        argv: Optional argument vector for testing; defaults to ``sys.argv``.

    Returns:
        A populated ``SPARQLPlotConfig`` instance.
    """

    parser = argparse.ArgumentParser(
        description="Run a SPARQL query on a TTL file and plot the results."
    )
    parser.add_argument(
        "--ttl",
        type=Path,
        default=Path("generated-rdf/extractions_chatgpt_4o.ttl"),
        help="Path to the TTL file containing the knowledge graph.",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=Path("queries/05-top10-problem-types.rq"),
        help="Path to the SPARQL query file.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Comma-separated variable names that supply bar labels. "
            "When omitted the script infers label columns."
        ),
    )
    parser.add_argument(
        "--value-column",
        default=None,
        help="Name of the variable that supplies bar heights; inferred when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("query_bar_chart.png"),
        help="Where to save the resulting bar chart image.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Display only the top N rows from the result set (0 keeps all rows).",
    )
    parser.add_argument(
        "--no-prettify-labels",
        action="store_true",
        help="Disable automatic prettifying of URI labels.",
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=18,
        help="Maximum character width before wrapping labels to a new line (0 disables wrapping).",
    )
    parser.add_argument(
        "--style",
        default="seaborn-v0_8-whitegrid",
        help="Matplotlib style to apply (use 'none' to skip).",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.0,
        help="Discard any rows whose value is below this threshold.",
    )
    parser.add_argument(
        "--orientation",
        choices=["vertical", "horizontal"],
        default="horizontal",
        help="Display bars vertically or horizontally.",
    )
    parser.add_argument(
        "--title",
        default="SPARQL Query Results",
        help="Title text to display above the chart.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI to use when saving the chart image.",
    )
    args = parser.parse_args(argv)
    label_columns = None
    if args.label_column:
        label_columns = tuple(
            column.strip() for column in str(args.label_column).split(",") if column.strip()
        )
    return SPARQLPlotConfig(
        ttl_path=args.ttl,
        query_path=args.query,
        label_columns=label_columns,
        value_column=args.value_column,
        output_path=args.output,
        top_n=args.top_n,
        prettify_labels=not args.no_prettify_labels,
        wrap_width=args.wrap_width,
        style=None if args.style.lower() == "none" else args.style,
        min_value=args.min_value,
        orientation=args.orientation,
        title=args.title,
        dpi=args.dpi,
    )


def load_query_text(query_path: Path) -> str:
    """Load the SPARQL query text from disk.

    Args:
        query_path: Path to the SPARQL query file.

    Returns:
        The raw SPARQL query string.
    """

    LOGGER.info("Loading SPARQL query from %s", query_path)
    return query_path.read_text(encoding="utf-8")


def run_query(ttl_path: Path, query_text: str) -> list[ResultRow]:
    """Execute the SPARQL query on the given TTL graph.

    Args:
        ttl_path: Path to the TTL knowledge graph.
        query_text: Query string to run.

    Returns:
        A list of SPARQL ``ResultRow`` objects.
    """

    LOGGER.info("Parsing TTL graph from %s", ttl_path)
    graph = Graph()
    graph.parse(ttl_path)
    LOGGER.info("Executing query against graph with %d triples", len(graph))
    results = list(graph.query(query_text))
    LOGGER.info("Query returned %d rows", len(results))
    return results


def ordered_columns(rows: Sequence[ResultRow]) -> list[str]:
    """Return query variable names ordered by their appearance in the result set."""

    if not rows:
        return []
    first_row = rows[0]
    ordered_pairs = sorted(first_row.labels.items(), key=lambda item: item[1])
    return [name for name, _ in ordered_pairs]


def literal_to_python(value: object) -> object:
    """Convert rdflib literal-like values into their Python equivalent."""

    if hasattr(value, "toPython"):
        return value.toPython()
    return value


def can_cast_to_float(value: object) -> bool:
    """Return True when the provided value can be cast to a float."""

    candidate = literal_to_python(value)
    try:
        float(candidate)
    except (TypeError, ValueError):
        return False
    return True


def is_numeric_column(rows: Sequence[ResultRow], column: str) -> bool:
    """Check whether every value in the column can be interpreted as numeric."""

    try:
        return all(can_cast_to_float(row[column]) for row in rows)
    except KeyError:
        return False


def choose_value_column(numeric_columns: Sequence[str]) -> str:
    """Pick the most likely numeric metric column."""

    if not numeric_columns:
        message = "Unable to find a numeric column suitable for plotting."
        raise ValueError(message)
    priority_tokens = ("count", "freq", "frequency", "total", "num", "value", "score", "sum")
    for column in numeric_columns:
        lower_name = column.lower()
        if any(token in lower_name for token in priority_tokens):
            return column
    return numeric_columns[0]


def infer_columns(
    rows: Sequence[ResultRow],
    configured_label_columns: tuple[str, ...] | None,
    configured_value_column: str | None,
) -> tuple[list[str], str]:
    """Infer which columns should be used for labels and values."""

    if not rows:
        message = "Cannot infer columns from an empty result set."
        raise ValueError(message)
    available_columns = ordered_columns(rows)
    numeric_columns = [column for column in available_columns if is_numeric_column(rows, column)]
    value_column = configured_value_column or choose_value_column(numeric_columns)
    if value_column not in available_columns:
        message = f"Requested value column '{value_column}' is not present in the results."
        raise ValueError(message)
    if configured_label_columns:
        missing = [column for column in configured_label_columns if column not in available_columns]
        if missing:
            message = f"Requested label columns {missing} are not present in the results."
            raise ValueError(message)
        label_columns = list(configured_label_columns)
    else:
        label_columns = [
            column for column in available_columns if column != value_column and column not in numeric_columns
        ]
        if not label_columns:
            fallback_candidates = [column for column in available_columns if column != value_column]
            if not fallback_candidates:
                message = (
                    "Unable to determine label columns automatically; only a numeric column was returned."
                )
                raise ValueError(message)
            label_columns = fallback_candidates
    LOGGER.info("Using label columns: %s; value column: %s", label_columns, value_column)
    return label_columns, value_column


def extract_columns(
    rows: Sequence[ResultRow], label_columns: Sequence[str], value_column: str
) -> tuple[list[str], list[float]]:
    """Extract the label and value columns from the query results.

    Args:
        rows: Result rows returned from the query.
        label_columns: Names of the label variables to combine.
        value_column: Name of the value variable to extract.

    Returns:
        Tuple containing the labels and values.
    """

    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        try:
            raw_value = row[value_column]
        except KeyError as exc:
            message = f"Column {exc} missing from result row: available={row.labels}"
            raise KeyError(message) from exc
        label_fragments: list[str] = []
        for column in label_columns:
            try:
                raw_label = row[column]
            except KeyError as exc:
                message = f"Column {exc} missing from result row: available={row.labels}"
                raise KeyError(message) from exc
            text_value = literal_to_python(raw_label)
            label_fragments.append(str(text_value))
        label_text = " • ".join(fragment for fragment in label_fragments if fragment)
        value_number = literal_to_python(raw_value)
        try:
            numeric_value = float(value_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Value column {value_column} must be numeric; got {value_number}") from exc
        labels.append(label_text)
        values.append(numeric_value)
    return labels, values


def limit_rows(
    labels: list[str], values: list[float], top_n: int
) -> tuple[list[str], list[float]]:
    """Limit the number of rows shown.

    Args:
        labels: Original labels.
        values: Original numeric values.
        top_n: Maximum number of rows to keep, or 0 for all.

    Returns:
        Possibly truncated labels and values.
    """

    if top_n <= 0 or top_n >= len(values):
        return labels, values
    return labels[:top_n], values[:top_n]


def prettify_label(label: str) -> str:
    """Convert URI or CamelCase identifiers into human-friendly labels."""

    def _prettify_single(raw_label: str) -> str:
        fragment = raw_label.split("#")[-1].split("/")[-1]
        cleaned = fragment.replace("_", " ").replace("-", " ")
        pretty = ""
        for index, char in enumerate(cleaned):
            if index > 0 and char.isupper() and cleaned[index - 1].islower():
                pretty += " "
            pretty += char
        return pretty.strip() or raw_label

    parts = [part.strip() for part in label.split("•")]
    prettified_parts = [_prettify_single(part) for part in parts if part]
    if not prettified_parts:
        return label
    return " • ".join(prettified_parts)


def maybe_prettify_labels(labels: list[str], enabled: bool) -> list[str]:
    """Apply prettification when enabled."""

    if not enabled:
        return labels
    return [prettify_label(label) for label in labels]


def maybe_wrap_labels(labels: list[str], width: int) -> list[str]:
    """Wrap labels to multiple lines when a width is provided."""

    if width <= 0:
        return labels
    wrapped: list[str] = []
    for label in labels:
        lines = textwrap.wrap(label, width=width) or [label]
        wrapped.append("\n".join(lines))
    return wrapped


def filter_min_value(
    labels: list[str], values: list[float], min_value: float
) -> tuple[list[str], list[float]]:
    """Filter out rows whose value is below the threshold."""

    if min_value <= 0:
        return labels, values
    filtered_pairs = [
        (label, value) for label, value in zip(labels, values, strict=True) if value >= min_value
    ]
    if not filtered_pairs:
        return [], []
    new_labels, new_values = map(list, zip(*filtered_pairs, strict=True))
    return new_labels, new_values


def sort_rows(labels: list[str], values: list[float]) -> tuple[list[str], list[float]]:
    """Sort rows from highest to lowest value."""

    ordered = sorted(zip(labels, values, strict=True), key=lambda pair: pair[1], reverse=True)
    sorted_labels, sorted_values = map(list, zip(*ordered, strict=True))
    return sorted_labels, sorted_values


def plot_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    output_path: Path,
    orientation: str,
    title: str,
    dpi: int,
) -> None:
    """Render and save a bar chart for the provided data.

    Args:
        labels: Labels to show on the x-axis.
        values: Numeric values for the bar heights.
        output_path: Destination path for the PNG image.
        orientation: Either ``"vertical"`` or ``"horizontal"``.
        title: Title text for the chart.
        dpi: Output resolution.
    """

    LOGGER.info("Plotting bar chart with %d bars", len(labels))
    if len(values) > 1:
        color_map = plt.colormaps.get_cmap("viridis")
        colors = color_map([i / (len(values) - 1) for i in range(len(values))])
    else:
        colors = ["#4477aa"]

    if orientation == "horizontal":
        figure_height = max(3.2, min(0.9 + 0.45 * len(labels), 9))
        figure_width = 9.5
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        bars = axis.barh(range(len(values)), values, color=colors)
        axis.set_xlabel("Count")
        axis.set_ylabel("Category")
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        for idx, (bar, value) in enumerate(zip(bars, values, strict=True)):
            axis.annotate(
                f"{value:.0f}",
                xy=(value, bar.get_y() + bar.get_height() / 2),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    else:
        figure_width = max(6.5, min(1.8 + len(labels) * 0.6, 12))
        figure, axis = plt.subplots(figsize=(figure_width, 5))
        bars = axis.bar(range(len(values)), values, color=colors)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_xlabel("Category")
        axis.set_ylabel("Count")
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(
                f"{value:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_title(title)
    axis.margins(x=0.02)
    grid_axis = axis.xaxis if orientation == "horizontal" else axis.yaxis
    grid_axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if orientation == "vertical":
        axis.set_ylim(top=max(values) * 1.15 if values else 1.0)
    figure.tight_layout()
    LOGGER.info("Saving chart to %s", output_path)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for command-line execution."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    plt.style.use("default")
    config = parse_args(argv)
    if config.style:
        plt.style.use(config.style)
    query_text = load_query_text(config.query_path)
    rows = run_query(config.ttl_path, query_text)
    if not rows:
        LOGGER.warning("Query returned no rows; skipping chart generation.")
        return
    label_columns, value_column = infer_columns(
        rows, config.label_columns, config.value_column
    )
    labels, values = extract_columns(rows, label_columns, value_column)
    labels = maybe_prettify_labels(labels, config.prettify_labels)
    labels, values = sort_rows(labels, values)
    labels, values = filter_min_value(labels, values, config.min_value)
    labels, values = limit_rows(labels, values, config.top_n)
    labels = maybe_wrap_labels(labels, config.wrap_width)
    if not labels:
        LOGGER.warning(
            "No rows remain after applying min-value/top-n filters; skipping chart generation."
        )
        return
    plot_bar_chart(
        labels,
        values,
        config.output_path,
        orientation=config.orientation,
        title=config.title,
        dpi=config.dpi,
    )


if __name__ == "__main__":
    main()
