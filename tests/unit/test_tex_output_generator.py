"""Tests for ExperimentTexGenerator — all offline, no LLM, no subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aras.experiments.tex_output_generator import ExperimentTexGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experiment_structure(
    tmp_path: Path,
    *,
    n_experiments: int = 3,
    with_figures: bool = True,
    with_results: bool = True,
    domain: str = "general_ml",
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create a minimal experiment directory structure for testing."""
    bundle_dir = tmp_path / "test-topic"
    bundle_dir.mkdir()

    experiments = []
    runs = []
    for i in range(1, n_experiments + 1):
        exp_name = f"exp{i}_test_experiment"
        exp_dir = bundle_dir / exp_name
        exp_dir.mkdir()

        # Create experiment script (minimal)
        (exp_dir / f"{exp_name}.py").write_text("print('hello')", encoding="utf-8")

        # Per-experiment results.json
        if with_results:
            per_results = {
                "accuracy": 0.85 + i * 0.02,
                "f1_score": 0.82 + i * 0.03,
                "wall_seconds": 42.0 + i * 10,
            }
            (exp_dir / "results.json").write_text(
                json.dumps(per_results), encoding="utf-8"
            )

        # Figures
        if with_figures:
            (exp_dir / "loss.png").write_bytes(b"FAKE_PNG")
            figs_dir = exp_dir / "figures"
            figs_dir.mkdir()
            (figs_dir / f"fig{i}.png").write_bytes(b"FAKE_PNG")

        experiments.append({
            "name": exp_name,
            "file": f"{exp_name}.py",
            "domain": domain,
            "dataset": "synthetic",
            "expected_metrics": ["accuracy"],
            "estimated_runtime_s": 60.0,
        })

        runs.append({
            "name": exp_name,
            "exit_code": 0,
            "wall_seconds": 42.0 + i * 10,
            "peak_rss_mb": 128.0,
            "stdout": "",
            "stderr": "",
            "artifacts": {},
            "metrics": {"accuracy": 0.85 + i * 0.02, "f1_score": 0.82 + i * 0.03},
        })

    manifest = {
        "topic": "Test Topic",
        "slug": "test-topic",
        "domain": domain,
        "experiments": experiments,
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    aggregate_results = {"slug": "test-topic", "runs": runs}
    (bundle_dir / "results.json").write_text(
        json.dumps(aggregate_results, indent=2), encoding="utf-8"
    )

    return bundle_dir, manifest, aggregate_results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExperimentTexGenerator:

    def test_tex_output_dir_created(self, tmp_path: Path) -> None:
        gen = ExperimentTexGenerator(bundle_dir=tmp_path)
        assert (tmp_path / "tex_output").exists()
        assert (tmp_path / "tex_output").is_dir()

    def test_generate_all_creates_tex_files(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex_dir = bundle_dir / "tex_output"
        for exp in manifest["experiments"]:
            tex_file = tex_dir / f"{exp['name']}.tex"
            assert tex_file.exists(), f"Expected {tex_file.name} to exist"

    def test_combined_tex_has_input_commands(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        combined = gen.generate_all(manifest=manifest, aggregate_results=results)

        content = combined.read_text(encoding="utf-8")
        assert "\\input{" in content
        assert combined.name == "experiments_combined.tex"

    def test_experiment_tex_has_subsection(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\subsection{" in tex

    def test_experiment_tex_has_table(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\begin{table}" in tex
        assert "\\toprule" in tex

    def test_experiment_tex_has_figure_reference(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\includegraphics" in tex

    def test_figure_path_uses_forward_slashes(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        # Extract \includegraphics lines
        for line in tex.splitlines():
            if "\\includegraphics" in line:
                # The path inside {} should have forward slashes only
                assert "\\\\" not in line.split("{")[-1], \
                    f"Backslash found in figure path: {line}"

    def test_figure_path_is_relative(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        for line in tex.splitlines():
            if "\\includegraphics" in line:
                # Extract path from \includegraphics[...]{PATH}
                path_part = line.split("{")[-1].split("}")[0]
                # Should not be an absolute path
                assert not path_part.startswith("/"), f"Absolute path found: {path_part}"
                assert ":" not in path_part[:3], f"Windows absolute path found: {path_part}"

    def test_metrics_table_from_dict(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        # Should contain the accuracy metric value
        assert "0.87" in tex or "Accuracy" in tex

    def test_comparison_table_from_list(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        # Add comparison_table to per-experiment results
        comp = [
            {"method": "Baseline", "accuracy": 0.80, "f1": 0.78},
            {"method": "Ours", "accuracy": 0.92, "f1": 0.90},
        ]
        exp_dir = bundle_dir / "exp1_test_experiment"
        per_results = json.loads((exp_dir / "results.json").read_text())
        per_results["comparison_table"] = comp
        (exp_dir / "results.json").write_text(json.dumps(per_results), encoding="utf-8")

        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "Baseline" in tex

    def test_latex_escape_applied_to_narrative(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        analysis = {"narrative": "Results show 50% improvement & reduction in cost $ value."}

        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results, analysis=analysis)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\&" in tex
        assert "\\$" in tex
        assert "\\%" in tex

    def test_timeout_safe_note_included(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        # Add TIMEOUT_SAFE to per-experiment results
        exp_dir = bundle_dir / "exp1_test_experiment"
        per_results = json.loads((exp_dir / "results.json").read_text())
        per_results["TIMEOUT_SAFE"] = True
        per_results["wall_seconds"] = 280.0
        (exp_dir / "results.json").write_text(json.dumps(per_results), encoding="utf-8")

        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "partial results" in tex

    def test_generate_all_with_paper_dir(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=1)
        paper_dir = tmp_path / "paper"
        paper_dir.mkdir()

        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(
            manifest=manifest,
            aggregate_results=results,
            paper_dir=paper_dir,
        )

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        # Figure paths should be relative to paper_dir, not tex_output_dir
        for line in tex.splitlines():
            if "\\includegraphics" in line:
                path_part = line.split("{")[-1].split("}")[0]
                # Should point up from paper/ into test-topic/exp1_.../
                assert "test-topic" in path_part or "exp1" in path_part

    def test_missing_figures_graceful(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(
            tmp_path, n_experiments=1, with_figures=False
        )
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        gen.generate_all(manifest=manifest, aggregate_results=results)

        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\includegraphics" not in tex

    def test_missing_results_json_graceful(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(
            tmp_path, n_experiments=1, with_results=False
        )
        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        combined = gen.generate_all(manifest=manifest, aggregate_results=results)

        assert combined.exists()
        tex = (bundle_dir / "tex_output" / "exp1_test_experiment.tex").read_text(encoding="utf-8")
        assert "\\subsection{" in tex

    def test_combined_tex_input_paths_relative_to_paper_dir(self, tmp_path: Path) -> None:
        bundle_dir, manifest, results = _make_experiment_structure(tmp_path, n_experiments=2)
        paper_dir = tmp_path / "paper"
        paper_dir.mkdir()

        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        combined = gen.generate_all(
            manifest=manifest,
            aggregate_results=results,
            paper_dir=paper_dir,
        )

        content = combined.read_text(encoding="utf-8")
        # Should have 2 \input{} lines with relative paths from paper_dir
        input_count = content.count("\\input{")
        assert input_count == 2, f"Expected 2 \\input lines, got {input_count}"
        # No backslashes in paths
        for line in content.splitlines():
            if "\\input{" in line:
                path_part = line.split("{")[1].split("}")[0]
                assert "\\\\" not in path_part

    def test_empty_manifest_produces_empty_combined(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "empty-topic"
        bundle_dir.mkdir()
        manifest: dict[str, Any] = {"topic": "Empty", "domain": "general_ml", "experiments": []}
        results: dict[str, Any] = {"runs": []}

        gen = ExperimentTexGenerator(bundle_dir=bundle_dir)
        combined = gen.generate_all(manifest=manifest, aggregate_results=results)

        content = combined.read_text(encoding="utf-8")
        assert "\\input{" not in content
