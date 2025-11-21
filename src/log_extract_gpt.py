"""LLM-assisted extraction of maintenance problems and actions via makeprov rules."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
from makeprov import InPath, OutPath, rule

LOGGER = logging.getLogger(__name__)
BASE_IRI = "https://w3id.org/zorro#"

PROBLEM_PROMPT = """Given a set of aircraft maintenance problem descriptions, output a json object with an array of problem objects: {"problems":[ problem1, problem2, ... ]}
Each problem is an object with the following fields:
- "id" (id of problem, first number in description)
- "engine" (where the problem occurs. Single-letter string ("R" or "L") or null)
- "cylinders" (array of integers, where the problem occurs, e.g. [1]. If "all" is mentioned, output [1,2,3,4])
- "part" (which single part causes the problem, e.g. "ENGINE BAFFLE")
- "problem" (ONE single word of what is going on, e.g. "LEAK", "FELL_OFF", "CRACK", "DAMAGE", "COMPRESSION", "NEED")
- "details" (more info that doesn't fit in the other fields, e.g. "BADLY")
The values of the fields MUST be substrings (spans) in the problem description.
All fields are optional except for "id".
If there is no very clear single part identified, don't fill in the field!
"""

ACTION_PROMPT = """Given a set of aircraft maintenance action descriptions, output a json object with an array of action objects: {"actions":[ action1, action2, ... ]}
Each action is an object with the following fields:
- "id" (id of action, first number in description)
- "engine" (where the action occurs. Single-letter string ("R" or "L") or null)
- "cylinders" (array of integers, where the action occurs, e.g. [1]. If "all" is mentioned, output [1,2,3,4])
- "part" (which single part causes the action, e.g. "ENGINE BAFFLE")
- "action" (ONE single word of what was done, e.g. "REPLACED", "TIGHTENED", "INSTALLED")
- "details" (more info that doesn't fit in the other fields, e.g. "NEW", "LEAK CHECK GOOD")
The values of the fields MUST be substrings (spans) in the action description.
All fields are optional except for "id".
If there is no very clear single part identified, don't fill in the field!
"""


@dataclass(slots=True)
class ChatGPTExtractionConfig:
    """Runtime configuration for ChatGPT-powered extractions."""

    model: str = "gpt-4o-mini"
    chunk_size: int = 20
    temperature: float = 0.0
    cache_dir: Path = Path("gpt-cache")
    cache_only: bool = True
    api_key_path: Path = Path("openai-key.txt")

    def ensure_cache_dir(self) -> None:
        """Create the cache directory if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_api_key(self) -> str:
        """Read the OpenAI API key from disk."""
        if not self.api_key_path.exists():
            msg = f"OpenAI key not found at {self.api_key_path}"
            raise FileNotFoundError(msg)
        return self.api_key_path.read_text(encoding="utf-8").strip()


def _clean_logs(path: InPath) -> pd.DataFrame:
    """Load and normalise aircraft maintenance logs."""
    logs = pd.read_csv(path)
    logs.columns = [c.lower() for c in logs.columns]
    for column in ("problem", "action"):
        logs[column] = logs[column].astype(str).str.strip(".").str.strip()
    logs["ident"] = logs["ident"].astype(str)
    return logs


def _chunk_text(
    frame: pd.DataFrame,
    column: str,
    chunk_size: int,
) -> Iterator[tuple[int, list[str]]]:
    """Yield chunks of concatenated IDENT + text strings for the chosen column."""
    subset = frame[["ident", column]].dropna()
    subset[column] = subset[column].astype(str).str.strip()
    subset = subset[subset[column].ne("")]

    rows = subset.to_dict("records")
    for offset in range(0, len(rows), chunk_size):
        batch = rows[offset : offset + chunk_size]
        lines = [f'{row["ident"]} {row[column]}' for row in batch]
        yield offset, lines


def _call_openai(messages: Sequence[dict[str, str]], config: ChatGPTExtractionConfig) -> str:
    """Call the OpenAI API and return the JSON content string."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(
            "openai package is required to call the API. Install openai>=1.0."
        ) from exc

    client = OpenAI(api_key=config.load_api_key())
    completion = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        response_format={"type": "json_object"},
        messages=list(messages),
    )
    return (completion.choices[0].message.content or "").strip()


def _load_cached_json(path: Path) -> dict:
    """Load cached JSON from disk, returning an empty object on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        msg = f"Cached response at {path} is not valid JSON"
        raise ValueError(msg) from exc


def _request_json_response(
    messages: Sequence[dict[str, str]],
    cache_path: Path,
    config: ChatGPTExtractionConfig,
) -> dict:
    """Return the JSON-decoded response, using cache when possible."""
    if cache_path.exists():
        LOGGER.debug("Using cached completion %s", cache_path)
        return _load_cached_json(cache_path)

    if config.cache_only:
        msg = (
            f"Missing cached completion {cache_path}. "
            "Set cache_only=False to regenerate with OpenAI."
        )
        raise FileNotFoundError(msg)

    LOGGER.info("Requesting new completion for %s", cache_path.name)
    response_text = _call_openai(messages, config)
    cache_path.write_text(response_text, encoding="utf-8")
    return json.loads(response_text or "{}")


def _extract_with_prompt(
    logs: pd.DataFrame,
    column: str,
    prompt: str,
    response_key: str,
    output_path: OutPath,
    cache_prefix: str,
    config: ChatGPTExtractionConfig,
) -> pd.DataFrame:
    """Extract structured data for the selected column using a ChatGPT prompt."""
    config.ensure_cache_dir()
    results: list[dict] = []
    for offset, lines in _chunk_text(logs, column, config.chunk_size):
        cache_path = config.cache_dir / f"{cache_prefix}-{offset:05d}.json"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n".join(lines)},
        ]
        response = _request_json_response(messages, cache_path, config)
        payload = response.get(response_key, [])
        if not isinstance(payload, list):
            LOGGER.warning(
                "Unexpected response structure for %s at chunk %s: %s",
                column,
                offset,
                response,
            )
            continue
        results.extend(payload)

    frame = pd.DataFrame.from_records(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


@rule(
    name="problem_extractions_chatgpt",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def extract_problems_chatgpt(
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    output_csv: OutPath = OutPath("log-extracted/problem_extractions_chatgpt_4o.csv"),
    *,
    config: ChatGPTExtractionConfig = ChatGPTExtractionConfig(),
) -> pd.DataFrame:
    """Extract structured problems from maintenance logs using ChatGPT."""
    LOGGER.info("Starting ChatGPT problem extraction (cache_only=%s)", config.cache_only)
    logs = _clean_logs(logs_csv)
    return _extract_with_prompt(
        logs=logs,
        column="problem",
        prompt=PROBLEM_PROMPT,
        response_key="problems",
        output_path=output_csv,
        cache_prefix="problem",
        config=config,
    )


@rule(
    name="action_extractions_chatgpt",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def extract_actions_chatgpt(
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    output_csv: OutPath = OutPath("log-extracted/action_extractions_chatgpt_4o.csv"),
    *,
    config: ChatGPTExtractionConfig = ChatGPTExtractionConfig(),
) -> pd.DataFrame:
    """Extract structured actions from maintenance logs using ChatGPT."""
    LOGGER.info("Starting ChatGPT action extraction (cache_only=%s)", config.cache_only)
    logs = _clean_logs(logs_csv)
    return _extract_with_prompt(
        logs=logs,
        column="action",
        prompt=ACTION_PROMPT,
        response_key="actions",
        output_path=output_csv,
        cache_prefix="action",
        config=config,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract_problems_chatgpt()
    extract_actions_chatgpt()
