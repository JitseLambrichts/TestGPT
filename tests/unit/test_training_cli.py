from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_export_training_console_script_points_to_clickhouse_exporter() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'imbalance-export-training = "imbalance_pipeline.training.export_clickhouse:main"'
        in pyproject
    )


def test_csv_training_console_script_and_make_workflow_are_available() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert 'imbalance-export-csv = "imbalance_pipeline.training.export_csv:main"' in pyproject
    assert "train-csv:" in makefile
    assert "imbalance-export-csv" in makefile
    assert "imbalance-train" in makefile


def test_make_export_training_requires_utc_window_and_forwards_arguments() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "export-training" in makefile
    assert 'test -n "$(START)" && test -n "$(END)" && test -n "$(OUTPUT)"' in makefile
    assert '"$(START)"' in makefile and '"$(END)"' in makefile and '"$(OUTPUT)"' in makefile


def test_training_workflow_documents_backfill_export_train_and_manual_promotion() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "README.md", ROOT / "docs/runbook.md")
    )
    markers = (
        "ODS133",
        "make export-training",
        "make train",
        "promote_bundle",
        "docker compose restart predictor api",
    )
    for marker in markers:
        assert marker in text
