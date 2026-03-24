from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "aras.experiments.templates.nlp_experiments",
        "aras.experiments.templates.cv_experiments",
        "aras.experiments.templates.rl_experiments",
        "aras.experiments.templates.general_ml_experiments",
        "aras.experiments.templates.theory_experiments",
    ],
)
def test_templates_export_experiments_list(module_name: str) -> None:
    m = importlib.import_module(module_name)
    assert hasattr(m, "EXPERIMENTS")
    exps = getattr(m, "EXPERIMENTS")
    assert isinstance(exps, list)
    assert len(exps) == 3
    names = [e["name"] for e in exps]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(
    "module_name",
    [
        "aras.experiments.templates.nlp_experiments",
        "aras.experiments.templates.cv_experiments",
        "aras.experiments.templates.rl_experiments",
        "aras.experiments.templates.general_ml_experiments",
        "aras.experiments.templates.theory_experiments",
    ],
)
def test_template_code_compiles_and_contains_contract(module_name: str) -> None:
    m = importlib.import_module(module_name)
    exps = m.EXPERIMENTS
    for e in exps:
        for k in ("name", "code", "dataset", "expected_metrics"):
            assert k in e
        code = e["code"]
        assert isinstance(code, str) and code.strip()
        compile(code, "<exp>", "exec")
        assert "METRIC_JSON" in code
        assert "results.json" in code
        assert ".aras_datasets" in code
        assert ("signal" in code) or ("threading.Timer" in code)


@pytest.mark.slow
def test_general_ml_exp1_runs_to_completion(tmp_path):
    """
    Mini-e2e: execute one real experiment end-to-end as a subprocess.

    Uses the fastest template (general_ml exp1) and asserts it produces
    required artifacts: results.json + loss.png.
    """
    import subprocess
    import sys

    m = importlib.import_module("aras.experiments.templates.general_ml_experiments")
    exp1 = m.EXPERIMENTS[0]
    code = exp1["code"]

    # Speed up for smoke on Windows/CPU: fewer estimators, fewer folds, smaller offline fallback.
    code = code.replace("cv_folds = 5", "cv_folds = 3")
    code = code.replace("n_estimators=100", "n_estimators=30")
    code = code.replace("max_iter=1000", "max_iter=300")
    # Avoid slow/hanging dataset fetches and heavy models: use only breast_cancer + logistic regression.
    code = code.replace(
        "    datasets, ds_source = _load_datasets()",
        "    datasets, ds_source = _load_datasets()\n"
        "    # Keep this mini-e2e fast and offline-friendly.\n"
        "    if isinstance(datasets, dict) and 'breast_cancer' in datasets:\n"
        "        datasets = {'breast_cancer': datasets['breast_cancer']}\n",
    )
    code = code.replace('("rf_100", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),\n', "")
    code = code.replace('("gboost_100", GradientBoostingClassifier(n_estimators=100, random_state=42)),\n', "")
    # Prevent any potential network fetch hangs (california housing).
    code = code.replace(
        "from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing  # type: ignore",
        "from sklearn.datasets import load_breast_cancer, load_wine  # type: ignore",
    )
    code = code.replace("cal = fetch_california_housing()", 'raise Exception("skip fetch_california_housing in mini-e2e")')

    script = tmp_path / "exp1.py"
    script.write_text(code, encoding="utf-8")

    # Run with a hard timeout; this template is designed to complete quickly on CPU.
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), timeout=120, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    assert (tmp_path / "results.json").exists(), f"missing results.json; stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert (tmp_path / "loss.png").exists(), f"missing loss.png; stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
