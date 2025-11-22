from __future__ import annotations

from makeprov import InPath, OutPath

from log_extract_ner import (
    NERInferenceConfig,
    NERTrainingConfig,
    extract_problems_ner,
    train_ner_model,
)


def test_train_and_predict_ner(tmp_path):
    """Training and inference should run end-to-end on annotated data."""
    model_dir = tmp_path / "ner_model"
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    train_ner_model(
        annotated_csv=InPath("Annotated problems Aircraft Data.csv"),
        model_dir=OutPath(str(model_dir)),
        train_split_csv=OutPath(str(train_csv)),
        test_split_csv=OutPath(str(test_csv)),
        config=NERTrainingConfig(epochs=1, batch_size=8),
    )

    assert model_dir.exists()
    assert train_csv.exists()
    assert test_csv.exists()

    output_csv = tmp_path / "predictions.csv"
    results = extract_problems_ner(
        logs_csv=InPath("Aircraft_Annotation_DataFile.csv"),
        model_dir=InPath(str(model_dir)),
        output_csv=OutPath(str(output_csv)),
        config=NERInferenceConfig(),
    )

    assert not results.empty
    assert {"id", "text"}.issubset(results.columns)
