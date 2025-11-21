"""Regex-based extraction of maintenance problems and actions.

The legacy notebook code is refactored into two provenance-aware rules that
emit CSV outputs compatible with downstream linking. Patterns are kept intact,
but the logic is wrapped in functions so we can test and orchestrate them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from makeprov import InPath, OutPath, rule

LOGGER = logging.getLogger(__name__)

LOC_PAT = (
    r"(?:(?:L|R)/H ?(?:REAR )?(?:AFT )?(?:ENG(?:INE)?,? ?)?)?"
    r"(?:CYL(?:INDER)? ?)?(?:ALL )?(?:#?\d(?: ?. \d)*)?(?: CYL(?:INDER)? ?)?"
)
PROBLEM_PATTERN = re.compile(
    r"^"
    rf"(?P<loc1>{LOC_PAT})? ?"
    r"(?P<part>\w[ \w]+?)S?,? "
    rf"(?:ON )?(?P<loc2>{LOC_PAT})? ?"
    r"(?:IS |ARE |HAS |HAVE |APPEARS TO BE )?(?:A )?(?:POSSIBLE )?"
    r"(?:EVIDENCE OF )?(?:COMING )?(?:SEVERAL )?(?:SHOWS SIGNS OF )?"
    r"(?P<problem>(?:OIL )?LEAK(?:ING)?S?|LOOSE|TORN|CRACKED|BROKEN|DAMAGED?"
    r"|WORN|MISSING|BAD|SHEAR(?:ED)?|BROKE|STUCK|STICK(?:ING)?|DIRTY|DEAD|FAILED"
    r"|NEEDS?|.*COMPRESSION.*)"
    rf",? ?(?:ON )?(?P<loc3>{LOC_PAT})? ?"
    r"(?P<rest>.*)",
    flags=re.IGNORECASE,
)

ACTION_PATTERN = re.compile(
    r"^(?:REMOVED & )?(?:RE)?"
    r"(?P<action>REPLACED|TIGHTENED|SECURED|ATTACHED|FASTENED|TORQUED|CLEANED|STOP DRILLED) ?"
    r"(?P<location>(?:(?:L|R)/H (?:ENG )?)?(?:CYL ?)?(?:#?\d(?: ?. \d)*)(?: CYL ?)?)? ?"
    r"(?:W/ )?(?:NEW )?"
    r"(?P<part>[^,.]*?\w)S?"
    r"(?: W/ .*)?(?:[,.] .*)?$",
    flags=re.IGNORECASE,
)


def _clean_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize maintenance log columns."""
    logs = raw_logs.copy()
    logs.columns = [c.lower() for c in logs.columns]
    for column in ("problem", "action"):
        logs[column] = logs[column].astype(str).str.strip(".").str.strip()
    logs["ident"] = logs["ident"].astype(str)
    return logs


def _join_locations(row: pd.Series) -> str:
    """Join non-empty location fragments."""
    values = [value for value in row if isinstance(value, str) and value.strip()]
    return " ".join(values)


def _clean_location(text: str | float) -> str | None:
    """Remove boilerplate tokens from location text."""
    if not isinstance(text, str):
        return None
    cleaned = text
    for token in ["CYLINDER", "ENGINE", "CYL", "ENG", "#", ",", "&", "(", ")"]:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("ALL", "1 2 3 4")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _split_cylinders(location: pd.Series) -> pd.Series:
    """Extract cylinder numbers from a location column."""
    cylinders = (
        location.fillna("")
        .str.extractall(r"(\d)")
        .groupby(level=0)
        .agg(lambda x: " ".join(sorted(set(x))))
    )
    return cylinders


def _filter_parts(series: pd.Series, reference: pd.Series, tail_count: int) -> pd.Series:
    """Restrict extracted parts to acceptable head nouns."""
    head_counts = (
        series.dropna()
        .astype(str)
        .apply(lambda x: x.split()[-1] if x else "")
        .value_counts()
    )
    whitelist = set(reference.dropna().astype(str))
    whitelist.update(
        head_counts[head_counts.index.difference(whitelist)].sort_values().tail(
            tail_count
        ).index
    )

    def keep(value: str | float) -> str | None:
        if not isinstance(value, str):
            return None
        tokens = value.split()
        return value if tokens and tokens[-1] in whitelist else None

    return series.fillna("").apply(keep)


def _write_csv(frame: pd.DataFrame, destination: OutPath) -> None:
    """Persist dataframe with consistent encoding."""
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


@dataclass(slots=True)
class RegexExtractionConfig:
    """Configuration shared by both regex extraction rules."""

    part_whitelist_tail: int = 10
    include_rest_column: bool = True


@rule(
    name="problem_extractions_regex",
    base_iri="https://w3id.org/zorro#",
    prov_dir="generated-provenance",
)
def extract_problems_regex(
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    part_classes_tsv: InPath = InPath("prompt-extracted/part-classes.tsv"),
    output_csv: OutPath = OutPath("log-extracted/problem_extractions_regex.csv"),
    *,
    config: RegexExtractionConfig = RegexExtractionConfig(),
) -> pd.DataFrame:
    """Extract problem mentions using hand-crafted regular expressions.

    Args:
        logs_csv: Path to the aircraft maintenance log CSV.
        part_classes_tsv: TSV containing acceptable part head nouns.
        output_csv: Destination path for the extracted problems.
        config: Additional options controlling filtering behaviour.

    Returns:
        DataFrame containing regex-based problem extractions.
    """
    LOGGER.info("Loading logs from %s", logs_csv)
    logs = _clean_logs(pd.read_csv(logs_csv))

    LOGGER.info("Applying problem regex")
    extracted = logs["problem"].str.extract(PROBLEM_PATTERN)
    extracted["id"] = logs["ident"]
    extracted["location"] = extracted[["loc1", "loc2", "loc3"]].apply(
        _join_locations, axis=1
    )
    extracted["location"] = extracted["location"].apply(_clean_location)
    extracted["cylinders"] = _split_cylinders(extracted["location"])
    extracted["engine"] = (
        extracted["location"]
        .fillna("")
        .str.replace(r"\d", "", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

    LOGGER.info("Filtering part names")
    part_classes = pd.read_csv(part_classes_tsv, sep="\t")["Part"]
    extracted["part"] = _filter_parts(
        extracted["part"].str.upper(), part_classes, config.part_whitelist_tail
    )

    if not config.include_rest_column:
        extracted["rest"] = ""

    result = extracted[
        ["id", "part", "problem", "rest", "cylinders", "engine"]
    ].fillna("")

    LOGGER.info("Writing problem extractions to %s", output_csv)
    _write_csv(result, output_csv)
    return result


@rule(
    name="action_extractions_regex",
    base_iri="https://w3id.org/zorro#",
    prov_dir="generated-provenance",
)
def extract_actions_regex(
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    part_classes_tsv: InPath = InPath("prompt-extracted/part-classes.tsv"),
    output_csv: OutPath = OutPath("log-extracted/action_extractions_regex.csv"),
    *,
    config: RegexExtractionConfig = RegexExtractionConfig(include_rest_column=False),
) -> pd.DataFrame:
    """Extract action mentions using regular expressions.

    Args:
        logs_csv: Path to the aircraft maintenance log CSV.
        part_classes_tsv: TSV containing acceptable part head nouns.
        output_csv: Destination path for regex-based action extractions.
        config: Additional options controlling filtering behaviour.

    Returns:
        DataFrame containing regex-based action extractions.
    """
    LOGGER.info("Loading logs from %s", logs_csv)
    logs = _clean_logs(pd.read_csv(logs_csv))

    LOGGER.info("Applying action regex")
    extracted = logs["action"].str.extract(ACTION_PATTERN)
    extracted["id"] = logs["ident"]
    extracted["location"] = extracted["location"].apply(_clean_location)
    extracted["cylinders"] = _split_cylinders(extracted["location"])
    extracted["engine"] = (
        extracted["location"]
        .fillna("")
        .str.replace(r"\d", "", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

    LOGGER.info("Filtering part names")
    part_classes = pd.read_csv(part_classes_tsv, sep="\t")["Part"]
    extracted["part"] = _filter_parts(
        extracted["part"].str.upper(), part_classes, config.part_whitelist_tail
    )

    result = extracted[["id", "action", "part", "cylinders", "engine"]].fillna("")

    LOGGER.info("Writing action extractions to %s", output_csv)
    _write_csv(result, output_csv)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_problems_regex()
    extract_actions_regex()
