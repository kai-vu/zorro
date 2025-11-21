"""Train and run a spaCy NER model for maintenance problem extraction."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from makeprov import InPath, OutPath, rule
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
BASE_IRI = "https://w3id.org/zorro#"

ENTITY_COLUMNS = ("LOCATION", "PART", "TGGEDPROBLEM")


@dataclass(slots=True)
class NERTrainingConfig:
    """Configuration for training the spaCy NER model."""

    test_size: float = 0.2
    random_state: int = 42
    epochs: int = 30
    batch_size: int = 16
    dropout: float = 0.2
    language: str = "en"


@dataclass(slots=True)
class NERInferenceConfig:
    """Configuration for running inference with the spaCy NER model."""

    include_columns: Sequence[str] = ("LOCATION", "PART", "TGGEDPROBLEM")


def _load_annotated_data(path: InPath) -> pd.DataFrame:
    """Read the annotated CSV and normalise column names."""
    frame = pd.read_csv(path)
    frame.columns = [col.upper() for col in frame.columns]
    if "PROBLEM" not in frame.columns:
        msg = "Annotated dataset must contain a PROBLEM column"
        raise ValueError(msg)
    frame["PROBLEM"] = frame["PROBLEM"].astype(str)
    return frame


def _build_spacy_examples(
    texts: Iterable[str],
    annotations: Iterable[dict[str, list[tuple[int, int, str]]]],
    nlp,
) -> list:
    """Create spaCy Example objects from char index annotations."""
    from spacy.training import Example

    examples = []
    for text, ann in zip(texts, annotations, strict=True):
        entities = ann.get("entities", [])
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is None:
                LOGGER.debug("Skipping invalid span %s-%s (%s)", start, end, label)
                continue
            spans.append(span)
        if spans:
            examples.append(Example.from_dict(doc, {"entities": spans}))
    return examples


def _split_label_values(value: str) -> list[str]:
    """Split a cell value into potential entity mentions."""
    if ";" in value:
        parts = value.split(";")
    elif "," in value:
        parts = value.split(",")
    else:
        parts = [value]
    return [part.strip() for part in parts if part.strip()]


def _find_span(text: str, value: str, used: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Find a non-overlapping span of value inside text."""
    start = text.find(value)
    while start != -1:
        span = (start, start + len(value))
        if all(span[1] <= s or span[0] >= e for s, e in used):
            return span
        start = text.find(value, start + 1)
    return None


def _generate_annotations(frame: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Generate spaCy-style entity annotations from the dataframe."""
    texts: list[str] = []
    annotations: list[dict] = []
    for _, row in frame.iterrows():
        text = row["PROBLEM"]
        entities: list[tuple[int, int, str]] = []
        used_spans: list[tuple[int, int]] = []
        for column in ENTITY_COLUMNS:
            value = str(row.get(column, "")).strip()
            if not value or value in {"None", "nan"}:
                continue
            for fragment in _split_label_values(value):
                span = _find_span(text, fragment, used_spans)
                if span is None:
                    LOGGER.debug(
                        "Could not locate entity %s within text %s",
                        fragment,
                        row.get("IDENT", "unknown"),
                    )
                    continue
                used_spans.append(span)
                entities.append((span[0], span[1], column))
        if entities:
            texts.append(text)
            annotations.append({"entities": entities})
    return texts, annotations


@rule(
    name="ner_train_model",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def train_ner_model(
    annotated_csv: InPath = InPath("Annotated problems Aircraft Data.csv"),
    model_dir: OutPath = OutPath("models/ner_maintenance"),
    train_split_csv: OutPath = OutPath("log-extracted/ner_train.csv"),
    test_split_csv: OutPath = OutPath("log-extracted/ner_test.csv"),
    *,
    config: NERTrainingConfig = NERTrainingConfig(),
):
    """Train a spaCy NER model from annotated maintenance logs."""
    import spacy
    from spacy.util import minibatch

    LOGGER.info("Loading annotated dataset from %s", annotated_csv)
    data = _load_annotated_data(annotated_csv)
    data = data.dropna(subset=["PROBLEM"])

    train_df, test_df = train_test_split(
        data,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )
    Path(train_split_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(test_split_csv).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_split_csv, index=False)
    test_df.to_csv(test_split_csv, index=False)

    train_texts, train_annotations = _generate_annotations(train_df)
    test_texts, test_annotations = _generate_annotations(test_df)
    if not train_texts:
        raise ValueError("No valid training examples produced from annotated data.")

    nlp = spacy.blank(config.language)
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    for ann in train_annotations:
        for _, _, label in ann["entities"]:
            ner.add_label(label)

    train_examples = _build_spacy_examples(train_texts, train_annotations, nlp)
    optimizer = nlp.initialize(lambda: train_examples)
    LOGGER.info(
        "Starting NER training (%s epochs, batch size %s)",
        config.epochs,
        config.batch_size,
    )
    for epoch in range(config.epochs):
        random.shuffle(train_examples)
        losses = {}
        batches = minibatch(train_examples, size=config.batch_size)
        for batch in batches:
            nlp.update(batch, drop=config.dropout, sgd=optimizer, losses=losses)
        LOGGER.debug("Epoch %s losses: %s", epoch + 1, losses)

    eval_examples = _build_spacy_examples(test_texts, test_annotations, nlp)
    if eval_examples:
        scorer = nlp.evaluate(eval_examples)
        scores = scorer if isinstance(scorer, dict) else scorer.scores
        LOGGER.info("NER evaluation scores: %s", scores)
    else:
        LOGGER.warning("No evaluation examples created; skipping evaluation.")

    model_path = Path(model_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_path)
    LOGGER.info("Saved spaCy NER model to %s", model_path)
    return model_path


def _predict_entities(doc, labels: Sequence[str]) -> dict[str, str]:
    """Aggregate entity predictions by label."""
    result: dict[str, set[str]] = {label: set() for label in labels}
    for ent in doc.ents:
        if ent.label_ in result:
            result[ent.label_].add(ent.text)
    return {label: "; ".join(sorted(values)) for label, values in result.items()}


@rule(
    name="problem_extractions_ner",
    base_iri=BASE_IRI,
    prov_dir="generated-provenance",
)
def extract_problems_ner(
    logs_csv: InPath = InPath("Aircraft_Annotation_DataFile.csv"),
    model_dir: InPath = InPath("models/ner_maintenance"),
    output_csv: OutPath = OutPath("log-extracted/problem_extractions_ner.csv"),
    *,
    config: NERInferenceConfig = NERInferenceConfig(),
) -> pd.DataFrame:
    """Run the trained NER model on problem logs to extract entities."""
    import spacy

    LOGGER.info("Loading spaCy model from %s", model_dir)
    nlp = spacy.load(Path(model_dir))

    logs = pd.read_csv(logs_csv)
    logs.columns = [c.lower() for c in logs.columns]
    logs["ident"] = logs["ident"].astype(str)
    logs["problem"] = logs["problem"].fillna("").astype(str)

    records = []
    for _, row in logs.iterrows():
        problem_text = row["problem"].strip()
        if not problem_text:
            continue
        doc = nlp(problem_text)
        aggregated = _predict_entities(doc, config.include_columns)
        record = {"id": row["ident"], "text": problem_text}
        record.update({label.lower(): value for label, value in aggregated.items()})
        records.append(record)

    frame = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    LOGGER.info("Wrote NER extraction results to %s", output_csv)
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_path = train_ner_model()
    extract_problems_ner(model_dir=InPath(str(model_path)))
