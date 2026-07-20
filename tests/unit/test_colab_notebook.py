import json
from pathlib import Path


def test_colab_notebook_is_a_thin_safe_driver_in_the_required_order() -> None:
    notebook = json.loads(Path("notebooks/train_colab.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    headings = (
        "Environment check",
        "Repository install",
        "Dataset acquisition/upload",
        "Configuration",
        "GPU training",
        "Evaluation display",
        "ONNX export and validation",
        "Bundle download",
    )
    positions = [text.index(heading) for heading in headings]

    assert positions == sorted(positions)
    assert "train_ensemble" in text
    assert "validate_bundle" in text
    assert "class ImbalanceForecaster" not in text
    assert "API_KEY=" not in text
    assert "PASSWORD=" not in text
