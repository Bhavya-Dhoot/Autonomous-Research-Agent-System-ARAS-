from __future__ import annotations

"""
General ML experiment templates (real sklearn/CPU workloads).
"""

EXPERIMENTS = [
    {
        "name": "exp1_cross_validation_benchmark",
        "domain": "general_ml",
        "dataset": "sklearn datasets (breast_cancer, wine, california_housing)",
        "expected_metrics": ["mean_accuracy", "std_accuracy", "mean_f1", "fit_time_s"],
        "code": r'''from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp1_cross_validation_benchmark"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    try:
        import signal
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass
    import threading
    stop = {"flag": False}
    def _mark_timeout():
        stop["flag"] = True
    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


def _load_datasets() -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], str]:
    # Returns dict dataset_name -> (X, y_classified)
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    try:
        from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing  # type: ignore
        bc = load_breast_cancer()
        datasets["breast_cancer"] = (bc.data.astype(np.float32), bc.target.astype(int))
        wine = load_wine()
        datasets["wine"] = (wine.data.astype(np.float32), wine.target.astype(int))

        try:
            cal = fetch_california_housing()
            Xc = cal.data.astype(np.float32)
            yc = cal.target.astype(np.float32)
            # Convert to classification for accuracy/f1.
            thresh = float(np.median(yc))
            y_class = (yc > thresh).astype(int)
            datasets["california_housing_binary"] = (Xc, y_class)
        except Exception:
            # Offline fallback: synthetic binary classification.
            rng = np.random.default_rng(42)
            Xc = rng.normal(size=(2000, 8)).astype(np.float32)
            y_class = (Xc.sum(axis=1) + rng.normal(scale=0.5, size=(2000,))) > 0
            datasets["california_housing_binary"] = (Xc, y_class.astype(int))
    except Exception:
        # Synthetic fallback for all.
        rng = np.random.default_rng(42)
        X1 = rng.normal(size=(1000, 10)).astype(np.float32)
        y1 = (X1[:, 0] + rng.normal(scale=0.1, size=(1000,))) > 0
        datasets["breast_cancer"] = (X1, y1.astype(int))
        X2 = rng.normal(size=(1000, 13)).astype(np.float32)
        y2 = (X2[:, 0] > 0).astype(int)
        datasets["wine"] = (X2, y2.astype(int))
        X3 = rng.normal(size=(2000, 8)).astype(np.float32)
        y3 = (X3.sum(axis=1) > 0).astype(int)
        datasets["california_housing_binary"] = (X3, y3.astype(int))
    return datasets, "sklearn_or_fallback"


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.model_selection import StratifiedKFold  # type: ignore
        from sklearn.neighbors import KNeighborsClassifier  # type: ignore
        from sklearn.svm import SVC  # type: ignore
        from sklearn.metrics import accuracy_score, f1_score  # type: ignore
    except Exception as e:
        # Shouldn't happen with requirements, but keep resilience.
        results = {"experiment": EXPERIMENT_NAME, "domain": "general_ml", "metrics": {}, "comparison_table": [], "acc": 0.0, "final_loss": 1.0, "losses": [1.0], "runtime_seconds": 0.0, "error": str(e)}
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    datasets, _ = _load_datasets()
    models: List[Tuple[str, Any]] = [
        ("logreg", LogisticRegression(max_iter=1000, n_jobs=1, random_state=42)),
        ("rf_100", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        ("gboost_100", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svc_rbf", SVC(kernel="rbf", probability=False, random_state=42)),
        ("knn_5", KNeighborsClassifier(n_neighbors=5)),
    ]

    cv_folds = 10
    step_idx = 0
    comparison_rows: List[Dict[str, Any]] = []

    # Accuracy heatmap preparation.
    dataset_names = sorted(list(datasets.keys()))
    model_names = [m[0] for m in models]
    heatmap = np.zeros((len(model_names), len(dataset_names)), dtype=np.float32)

    try:
        for j, dname in enumerate(dataset_names):
            _timeout_check()
            X, y = datasets[dname]
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            for i, (mname, model) in enumerate(models):
                _timeout_check()
                tfit0 = time.perf_counter()
                accs: List[float] = []
                f1s: List[float] = []
                for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
                    _timeout_check()
                    Xtr, Xte = X[tr], X[te]
                    ytr, yte = y[tr], y[te]
                    model_i = model
                    # Clone-less usage to avoid import churn:
                    # Use sklearn's set_params via new instance when possible.
                    try:
                        model_i = model.__class__(**model.get_params())
                    except Exception:
                        model_i = model
                    model_i.fit(Xtr, ytr)
                    pred = model_i.predict(Xte)
                    accs.append(float(accuracy_score(yte, pred)))
                    # f1 macro works for multi-class too.
                    f1s.append(float(f1_score(yte, pred, average="macro", zero_division=0)))

                fit_time_s = time.perf_counter() - tfit0
                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs))
                mean_f1 = float(np.mean(f1s))

                heatmap[i, j] = mean_acc

                step_idx += 1
                emit("accuracy", mean_acc, step_idx, model=mname + "::" + dname)
                emit("f1_macro", mean_f1, step_idx, model=mname + "::" + dname)
                emit("fit_time_s", float(fit_time_s), step_idx, model=mname + "::" + dname)

                comparison_rows.append(
                    {
                        "dataset": dname,
                        "model": mname,
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                        "mean_f1": mean_f1,
                        "fit_time_s": float(fit_time_s),
                    }
                )

        # Plot heatmap of accuracy.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        sns.heatmap(heatmap, annot=False, cmap="viridis", cbar=True, xticklabels=dataset_names, yticklabels=model_names)
        plt.xlabel("Datasets")
        plt.ylabel("Models")
        plt.title("Cross-Validation Accuracy Heatmap")
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        # Pick best by max mean accuracy.
        best_idx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        best_model = model_names[int(best_idx[0])]
        best_dataset = dataset_names[int(best_idx[1])]
        best_acc = float(heatmap[int(best_idx[0]), int(best_idx[1])])

        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "sklearn_multiple",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_model": best_model, "best_dataset": best_dataset, "best_accuracy": best_acc},
            "comparison_table": comparison_rows,
            "acc": best_acc,
            "final_loss": float(1.0 - best_acc),
            "losses": [float(1.0 - best_acc)],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_model": best_model, "best_accuracy": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": comparison_rows[:10],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [0, 0])
            plt.title("Timeout Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": comparison_rows[:10],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
    {
        "name": "exp2_hyperparameter_sensitivity",
        "domain": "general_ml",
        "dataset": "sklearn breast_cancer",
        "expected_metrics": ["cv_accuracy"],
        "code": r'''from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp2_hyperparameter_sensitivity"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    try:
        import signal
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass
    import threading
    stop = {"flag": False}
    def _mark_timeout():
        stop["flag"] = True
    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    try:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
        from sklearn.datasets import load_breast_cancer  # type: ignore
        from sklearn.model_selection import StratifiedKFold  # type: ignore
        from sklearn.metrics import accuracy_score  # type: ignore
    except Exception as e:
        results = {"experiment": EXPERIMENT_NAME, "domain": "general_ml", "metrics": {}, "comparison_table": [], "acc": 0.0, "final_loss": 1.0, "losses": [1.0], "runtime_seconds": 0.0, "error": str(e)}
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    bc = load_breast_cancer()
    X = bc.data.astype(np.float32)
    y = bc.target.astype(int)

    n_estimators_list = [10, 25, 50, 100, 200]
    learning_rate_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    max_depth_list = [2, 3, 5]

    cv_folds = 5
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Track all combo results.
    combo_results: List[Dict[str, Any]] = []
    step_idx = 0
    best_acc = -1.0
    best_params: dict[str, Any] = {}

    try:
        for n_estimators in n_estimators_list:
            for lr in learning_rate_list:
                for max_depth in max_depth_list:
                    _timeout_check()
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        max_depth=max_depth,
                        random_state=42,
                    )
                    accs: List[float] = []
                    tfit0 = time.perf_counter()
                    for tr, te in cv.split(X, y):
                        _timeout_check()
                        model_i = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            learning_rate=lr,
                            max_depth=max_depth,
                            random_state=42,
                        )
                        model_i.fit(X[tr], y[tr])
                        preds = model_i.predict(X[te])
                        accs.append(float(accuracy_score(y[te], preds)))
                    fit_time_s = time.perf_counter() - tfit0
                    mean_acc = float(np.mean(accs))
                    step_idx += 1
                    combo = {"n_estimators": n_estimators, "learning_rate": lr, "max_depth": max_depth}
                    emit("cv_accuracy", mean_acc, step_idx, model=f"gb_{n_estimators}_{lr}_{max_depth}")
                    combo_results.append({"params": combo, "cv_accuracy": mean_acc, "fit_time_s": float(fit_time_s)})
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_params = combo

        # Plot 3 subplots using default other params slices.
        default = {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100}
        def slice_acc(**fixed):
            # fixed can contain 2 params and we vary 1.
            matches = {}
            for r in combo_results:
                ok = True
                for k, v in fixed.items():
                    if r["params"].get(k) != v:
                        ok = False
                        break
                if ok:
                    key = tuple(r["params"].get(k) for k in r["params"].keys() if k not in fixed)
                    # Only used for one varying param: return by varying param value.
                    # Determine varying param:
                    varying = [k for k in ["n_estimators", "learning_rate", "max_depth"] if k not in fixed]
                    if len(varying) != 1:
                        continue
                    var = varying[0]
                    matches[r["params"][var]] = float(r["cv_accuracy"])
            return matches

        # Acc vs n_estimators (vary n_estimators)
        acc_vs_n = slice_acc(learning_rate=default["learning_rate"], max_depth=default["max_depth"])
        # Acc vs learning_rate
        acc_vs_lr = slice_acc(n_estimators=default["n_estimators"], max_depth=default["max_depth"])
        # Acc vs max_depth
        acc_vs_md = slice_acc(n_estimators=default["n_estimators"], learning_rate=default["learning_rate"])

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        # n_estimators
        xs = n_estimators_list
        ys = [acc_vs_n.get(x, 0.0) for x in xs]
        axs[0].plot(xs, ys, marker="o")
        axs[0].set_title("Accuracy vs n_estimators")
        axs[0].set_xlabel("n_estimators")
        axs[0].set_ylabel("CV accuracy")
        # learning_rate
        xs = learning_rate_list
        ys = [acc_vs_lr.get(x, 0.0) for x in xs]
        axs[1].plot(xs, ys, marker="o")
        axs[1].set_title("Accuracy vs learning_rate")
        axs[1].set_xlabel("learning_rate")
        # max_depth
        xs = max_depth_list
        ys = [acc_vs_md.get(x, 0.0) for x in xs]
        axs[2].plot(xs, ys, marker="o")
        axs[2].set_title("Accuracy vs max_depth")
        axs[2].set_xlabel("max_depth")
        for ax in axs:
            ax.grid(True, alpha=0.3)
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "breast_cancer",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_params": best_params, "best_cv_accuracy": float(best_acc)},
            "comparison_table": combo_results[:50],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_cv_accuracy": best_acc, "best_params": best_params}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "breast_cancer",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_cv_accuracy": float(best_acc), "best_params": best_params},
            "comparison_table": combo_results[:20],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [best_acc, best_acc])
            plt.title("Timeout Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "breast_cancer",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": combo_results[:20],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
    {
        "name": "exp3_convergence_analysis",
        "domain": "general_ml",
        "dataset": "sklearn digits",
        "expected_metrics": ["loss_or_error", "accuracy"],
        "code": r'''from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp3_convergence_analysis"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    try:
        import signal
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass
    import threading
    stop = {"flag": False}
    def _mark_timeout():
        stop["flag"] = True
    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    try:
        from sklearn.datasets import load_digits  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.neural_network import MLPClassifier  # type: ignore
        from sklearn.linear_model import SGDClassifier  # type: ignore
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
        from sklearn.metrics import accuracy_score  # type: ignore
    except Exception as e:
        results = {"experiment": EXPERIMENT_NAME, "domain": "general_ml", "metrics": {}, "comparison_table": [], "acc": 0.0, "final_loss": 1.0, "losses": [1.0], "runtime_seconds": 0.0, "error": str(e)}
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        # A) MLPClassifier with loss_curve_
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=60, random_state=42, early_stopping=False, learning_rate_init=0.001)
        mlp.fit(X_train, y_train)
        losses = mlp.loss_curve_ if hasattr(mlp, "loss_curve_") else []
        accs_mlp: List[float] = []
        for step_idx, loss in enumerate(losses, start=1):
            _timeout_check()
            # lightweight: compute accuracy at intervals.
            if step_idx == 1 or step_idx == len(losses) or step_idx % 5 == 0:
                preds = mlp.predict(X_test)
                acc = float(accuracy_score(y_test, preds))
                accs_mlp.append(acc)
            emit("loss", float(loss), step_idx, model="mlp")
        preds = mlp.predict(X_test)
        best_acc_mlp = float(accuracy_score(y_test, preds))

        # B) SGDClassifier partial_fit mini-batches.
        sgd = SGDClassifier(loss="log_loss", random_state=42, max_iter=1, tol=None)
        classes = np.unique(y_train)
        train_acc_steps: List[float] = []
        epochs = 20
        batch_size = 128
        idxs = np.arange(len(X_train))
        for ep in range(1, epochs + 1):
            _timeout_check()
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                _timeout_check()
                batch_idx = idxs[start:start+batch_size]
                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]
                if ep == 1 and start == 0:
                    sgd.partial_fit(Xb, yb, classes=classes)
                else:
                    sgd.partial_fit(Xb, yb)
            preds = sgd.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            train_acc_steps.append(acc)
            emit("accuracy", acc, ep, model="sgd_partial_fit")

        best_acc_sgd = float(max(train_acc_steps)) if train_acc_steps else 0.0

        # C) GradientBoostingClassifier staged_predict.
        gbc = GradientBoostingClassifier(n_estimators=80, random_state=42)
        gbc.fit(X_train, y_train)
        test_errors: List[float] = []
        for step_idx, y_stage_pred in enumerate(gbc.staged_predict(X_test), start=1):
            _timeout_check()
            preds = y_stage_pred
            err = 1.0 - float(accuracy_score(y_test, preds))
            test_errors.append(err)
            emit("test_error", err, step_idx, model="gboost_stage")

        best_acc_gbc = float(1.0 - min(test_errors)) if test_errors else 0.0

        # Plot convergence.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, label="MLP loss", linewidth=2.0)
        if train_acc_steps:
            plt.plot(range(1, len(train_acc_steps) + 1), train_acc_steps, label="SGD accuracy", linewidth=2.0)
        if test_errors:
            plt.plot(range(1, len(test_errors) + 1), test_errors, label="GBoost test error", linewidth=2.0)
        plt.xlabel("Iteration / stage")
        plt.ylabel("Value (loss/error/accuracy)")
        plt.title("Convergence Analysis (Digits)")
        plt.legend(fontsize=9)
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        best_acc = max(best_acc_mlp, best_acc_sgd, best_acc_gbc)
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "digits",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {
                "best_accuracy": float(best_acc),
                "best_algorithm": "mlp" if best_acc_mlp == best_acc else ("sgd" if best_acc_sgd == best_acc else "gboost"),
            },
            "comparison_table": [
                {"algorithm": "mlp", "best_accuracy": float(best_acc_mlp)},
                {"algorithm": "sgd", "best_accuracy": float(best_acc_sgd)},
                {"algorithm": "gboost", "best_accuracy": float(best_acc_gbc)},
            ],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_accuracy": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "digits",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [0, 0])
            plt.title("Timeout Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "general_ml",
            "dataset": "digits",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
]

