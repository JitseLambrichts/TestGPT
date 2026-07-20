import subprocess
import sys
import zipfile
from pathlib import Path

DASHBOARD = (
    Path(__file__).parents[3]
    / "src"
    / "imbalance_pipeline"
    / "serving"
    / "dashboard"
)
INDEX = DASHBOARD / "index.html"
SCRIPT = DASHBOARD / "dashboard.js"
STYLES = DASHBOARD / "dashboard.css"
CHART = DASHBOARD / "vendor" / "chart.umd.min.js"
LICENSE = DASHBOARD / "vendor" / "LICENSE.md"
PROJECT_ROOT = Path(__file__).parents[3]


def test_dashboard_has_required_live_regions_and_cards() -> None:
    html = INDEX.read_text()

    for hook in (
        "live-status",
        "last-update",
        "model-version",
        "expected-value",
        "actual-value",
        "error-value",
        "flip-value",
        "state-value",
        "prediction-chart",
        "chart-summary",
        "history-body",
    ):
        assert f'id="{hook}"' in html
    assert 'aria-live="polite"' in html
    assert 'id="retry-button"' in html


def test_dashboard_table_contains_the_approved_exact_columns() -> None:
    html = INDEX.read_text()

    for heading in (
        "Doeltijd",
        "Voorspeld",
        "Werkelijk",
        "Afwijking",
        "Flipkans",
        "Flip voorspeld",
        "Flip werkelijk",
        "Resultaat",
    ):
        assert f">{heading}<" in html


def test_dashboard_javascript_uses_live_window_and_refresh_interval() -> None:
    javascript = SCRIPT.read_text()

    assert "15_000" in javascript
    assert "6 * 60 * 60 * 1000" in javascript
    assert "2 * 60 * 1000" in javascript
    assert 'timeZone: "Europe/Brussels"' in javascript
    assert "row.flip_actual === row.will_flip" in javascript
    assert "document.visibilityState" in javascript
    assert 'chart.update("none")' in javascript


def test_dashboard_preserves_pending_and_error_copy() -> None:
    javascript = SCRIPT.read_text()

    assert "In afwachting" in javascript
    assert "Verouderd" in javascript
    assert "Niet bereikbaar" in javascript
    assert "Nog geen live voorspellingen beschikbaar" in javascript


def test_dashboard_styles_cover_responsiveness_focus_and_reduced_motion() -> None:
    css = STYLES.read_text()

    assert "font-variant-numeric: tabular-nums" in css
    assert ":focus-visible" in css
    assert "@media (max-width:" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "overflow-x: auto" in css


def test_chartjs_is_local_pinned_and_licensed() -> None:
    html = INDEX.read_text()

    assert 'src="/dashboard-assets/vendor/chart.umd.min.js"' in html
    assert "https://" not in html
    assert CHART.stat().st_size > 100_000
    license_text = LICENSE.read_text()
    assert "Chart.js 4.5.1" in license_text
    assert "MIT License" in license_text
    assert "sha512-GIjfiT9" in license_text


def test_built_wheel_includes_dashboard_assets(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(PROJECT_ROOT),
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
    required = {
        "imbalance_pipeline/serving/dashboard/index.html",
        "imbalance_pipeline/serving/dashboard/dashboard.css",
        "imbalance_pipeline/serving/dashboard/dashboard.js",
        "imbalance_pipeline/serving/dashboard/vendor/chart.umd.min.js",
        "imbalance_pipeline/serving/dashboard/vendor/LICENSE.md",
    }

    assert required <= members


def test_readme_documents_live_dashboard_behavior() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()

    assert "http://localhost:8000/dashboard" in readme
    assert "15 seconden" in readme
    assert "uitsluitend live" in readme
