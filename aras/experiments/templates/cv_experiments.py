from __future__ import annotations

"""
Computer vision experiment templates.

CoderAgent will import this module and write each entry's `code`
into runnable `exp*.py` files inside the experiment bundle directory.
"""

EXPERIMENTS = [
    {
        "name": "exp1_image_classification",
        "domain": "computer_vision",
        "dataset": "MNIST (torchvision) / sklearn digits fallback",
        "expected_metrics": ["accuracy", "f1_macro", "train_time_s"],
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

from sklearn.metrics import accuracy_score, f1_score


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp1_image_classification"


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


def _load_dataset(n_train_limit: int, n_test_limit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    # Primary: torchvision MNIST.
    try:
        from torchvision.datasets import MNIST  # type: ignore
        from torchvision import transforms  # type: ignore

        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = MNIST(root=str(CACHE_DIR / "torchvision"), train=True, download=True, transform=transform)
        test_ds = MNIST(root=str(CACHE_DIR / "torchvision"), train=False, download=True, transform=transform)

        n_train = min(n_train_limit, len(train_ds))
        n_test = min(n_test_limit, len(test_ds))

        # `train_ds.data` is available as uint8.
        X_train = train_ds.data[:n_train].numpy().astype(np.float32) / 255.0
        y_train = train_ds.targets[:n_train].numpy().astype(int)
        X_test = test_ds.data[:n_test].numpy().astype(np.float32) / 255.0
        y_test = test_ds.targets[:n_test].numpy().astype(int)

        X_train = X_train.reshape(n_train, -1)
        X_test = X_test.reshape(n_test, -1)
        return X_train, y_train, X_test, y_test, "mnist"
    except Exception:
        pass

    # Fallback: sklearn digits (8x8).
    try:
        from sklearn.datasets import load_digits  # type: ignore

        digits = load_digits()
        X = digits.data.astype(np.float32)  # (n, 64)
        y = digits.target.astype(int)

        n_train = min(n_train_limit, X.shape[0])
        n_test = min(n_test_limit, X.shape[0] - n_train)
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train : n_train + n_test]
        y_test = y[n_train : n_train + n_test]
        return X_train, y_train, X_test, y_test, "digits_fallback"
    except Exception:
        pass

    # Final offline fallback: synthetic separable data.
    rng = np.random.default_rng(42)
    n_train = min(n_train_limit, 3000)
    n_test = min(n_test_limit, 1000)
    d = 64
    w = rng.normal(size=(10, d))
    y_train = rng.integers(0, 10, size=(n_train,))
    y_test = rng.integers(0, 10, size=(n_test,))
    X_train = w[y_train] + 0.5 * rng.normal(size=(n_train, d))
    X_test = w[y_test] + 0.5 * rng.normal(size=(n_test, d))
    return X_train.astype(np.float32), y_train.astype(int), X_test.astype(np.float32), y_test.astype(int), "synthetic_offline"


def _plot_learning_curve(train_sizes: List[int], curves: Dict[str, List[float]], title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
    plt.figure(figsize=(7, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(curves)))
    for (i, (name, ys)) in enumerate(curves.items()):
        plt.plot(train_sizes, ys, marker="o", linewidth=2.0, label=name, color=colors[i])
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout(pad=0.5)
    plt.savefig("loss.png", dpi=300)
    plt.savefig("loss.pdf")
    plt.close()


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    X_train, y_train, X_test, y_test, dataset_name = _load_dataset(5000, 1000)
    train_sizes_raw = [500, 1000, 2000, 5000]
    train_sizes: List[int] = [int(sz) for sz in train_sizes_raw if int(sz) <= int(X_train.shape[0])]
    if not train_sizes:
        train_sizes = [int(min(500, int(X_train.shape[0])))]

    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.svm import SVC  # type: ignore

    models = [
        ("mlp_128_64", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42, early_stopping=True)),
        ("rf_50", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)),
        ("svc_rbf_c1", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ]

    curves_accuracy: Dict[str, List[float]] = {name: [] for name, _ in models}
    best_model: str | None = None
    best_acc: float = -1.0
    best_f1: float = -1.0
    comparison_table: List[Dict[str, Any]] = []

    try:
        for step_idx, sz in enumerate(train_sizes, start=1):
            _timeout_check()
            Xtr = X_train[:sz]
            ytr = y_train[:sz]
            for model_name, model in models:
                _timeout_check()
                tfit0 = time.perf_counter()
                model.fit(Xtr, ytr)
                train_time_s = time.perf_counter() - tfit0
                preds = model.predict(X_test)
                acc = float(accuracy_score(y_test, preds))
                f1m = float(f1_score(y_test, preds, average="macro", zero_division=0))
                curves_accuracy[model_name].append(acc)
                emit("accuracy", acc, step_idx, model=model_name)
                emit("f1_macro", f1m, step_idx, model=model_name)
                emit("train_time_s", float(train_time_s), step_idx, model=model_name)
                comparison_table.append({"model": model_name, "train_size": int(sz), "accuracy": acc, "f1": f1m, "train_time_s": float(train_time_s)})
                if acc > best_acc:
                    best_acc = acc
                    best_f1 = f1m
                    best_model = model_name

        runtime_seconds = time.perf_counter() - t0
        _plot_learning_curve(train_sizes, curves_accuracy, f"Learning Curve ({dataset_name})")

        results = {
            "experiment": "exp1_image_classification",
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_model": best_model, "best_accuracy": float(best_acc), "best_f1_macro": float(best_f1)},
            "comparison_table": comparison_table[:20],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(a)) for a in (curves_accuracy.get(best_model or "", []) or [best_acc])],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": "exp1_image_classification", "best_model": best_model, "acc": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": "exp1_image_classification",
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_model": best_model, "best_accuracy": float(best_acc), "best_f1_macro": float(best_f1)},
            "comparison_table": comparison_table[:20],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [best_acc, best_acc])
            plt.title("Timeout Learning Curve Proxy")
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
            "experiment": "exp1_image_classification",
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": comparison_table[:20],
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
            "error": err[-2000:],
            "timed_out": False,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [0, 0])
            plt.title("Error Proxy")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
    {
        "name": "exp2_feature_extraction_benchmark",
        "domain": "computer_vision",
        "dataset": "Olivetti faces / digits upsample fallback",
        "expected_metrics": ["mean_cv_accuracy", "std_cv_accuracy", "fit_time_s"],
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
EXPERIMENT_NAME = "exp2_feature_extraction_benchmark"


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


def _load_faces(n_images_limit: int = 400) -> Tuple[np.ndarray, np.ndarray, str]:
    # Primary: olivetti faces.
    try:
        from sklearn.datasets import fetch_olivetti_faces  # type: ignore
        X, y = fetch_olivetti_faces(shuffle=True, random_state=42, download_if_missing=True, return_X_y=True)
        n = min(n_images_limit, X.shape[0])
        return X[:n].astype(np.float32), y[:n].astype(int), "olivetti_faces"
    except Exception:
        pass

    # Fallback: upsample digits to 64x64.
    try:
        from sklearn.datasets import load_digits  # type: ignore
        digits = load_digits()
        X = digits.data.astype(np.float32)  # n x 64
        y = digits.target.astype(int)
        n = min(n_images_limit, X.shape[0])
        X = X[:n]
        y = y[:n]
        X_2d = X.reshape(n, 8, 8)
        # 8x8 -> 64x64 (repeat by 8)
        X_big = np.kron(X_2d, np.ones((8, 8), dtype=np.float32))
        return X_big.reshape(n, -1), y, "digits_upsampled"
    except Exception:
        pass

    rng = np.random.default_rng(42)
    n = min(n_images_limit, 320)
    X = rng.normal(size=(n, 4096)).astype(np.float32)
    y = rng.integers(0, 10, size=(n,)).astype(int)
    return X, y, "synthetic_offline"


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    X, y, dataset_name = _load_faces(400)
    X = (X - X.mean()) / (X.std() + 1e-6)

    from sklearn.decomposition import FastICA, NMF, PCA  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import KFold, cross_val_score  # type: ignore

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    methods = [
        ("raw_pixels", None),
        ("pca_50", PCA(n_components=50, random_state=42)),
        ("nmf_50", NMF(n_components=50, random_state=42, init="nndsvda", max_iter=300)),
        ("ica_50", FastICA(n_components=50, random_state=42, max_iter=400)),
    ]

    out_table: List[Dict[str, Any]] = []
    try:
        for step_idx, (name, transformer) in enumerate(methods, start=1):
            _timeout_check()
            t0_fit = time.perf_counter()
            Xtr = X
            if transformer is not None:
                Xtr = transformer.fit_transform(X)
            scores = cross_val_score(clf, Xtr, y, cv=kf, scoring="accuracy", n_jobs=1)
            fit_time_s = time.perf_counter() - t0_fit
            mean_acc = float(np.mean(scores))
            std_acc = float(np.std(scores))
            emit("mean_cv_accuracy", mean_acc, step_idx, model=name)
            emit("std_cv_accuracy", std_acc, step_idx, model=name)
            emit("fit_time_s", float(fit_time_s), step_idx, model=name)
            out_table.append({"method": name, "mean_cv_accuracy": mean_acc, "std_cv_accuracy": std_acc, "fit_time_s": float(fit_time_s)})

        # Plot accuracy with error bars.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        names = [r["method"] for r in out_table]
        means = [r["mean_cv_accuracy"] for r in out_table]
        stds = [r["std_cv_accuracy"] for r in out_table]
        x = np.arange(len(names))
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        plt.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.9)
        plt.xticks(x, names, rotation=20, ha="right", fontsize=9)
        plt.ylabel("CV Accuracy")
        plt.title(f"Feature Extraction Benchmark ({dataset_name})")
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        best_idx = int(np.argmax([r["mean_cv_accuracy"] for r in out_table])) if out_table else 0
        best_mean = float(max([r["mean_cv_accuracy"] for r in out_table])) if out_table else 0.0

        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_method": out_table[best_idx]["method"] if out_table else None, "best_mean_cv_accuracy": best_mean},
            "comparison_table": out_table,
            "acc": best_mean,
            "final_loss": float(1.0 - best_mean),
            "losses": [float(1.0 - r["mean_cv_accuracy"]) for r in out_table] if out_table else [1.0],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_method": results["metrics"]["best_method"]}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        best_mean = float(max([r["mean_cv_accuracy"] for r in out_table])) if out_table else 0.0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": out_table,
            "acc": best_mean,
            "final_loss": float(1.0 - best_mean),
            "losses": [float(1.0 - best_mean)],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [best_mean, best_mean])
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
        best_mean = float(max([r["mean_cv_accuracy"] for r in out_table])) if out_table else 0.0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": out_table,
            "acc": best_mean,
            "final_loss": float(1.0 - best_mean),
            "losses": [float(1.0 - best_mean)],
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
        "name": "exp3_data_augmentation_ablation",
        "domain": "computer_vision",
        "dataset": "sklearn digits + numpy/scipy augmentations",
        "expected_metrics": ["test_accuracy", "train_size_after_aug"],
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
EXPERIMENT_NAME = "exp3_data_augmentation_ablation"


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


def _load_digits() -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.datasets import load_digits  # type: ignore
        digits = load_digits()
        return digits.data.astype(np.float32), digits.target.astype(int), "digits"
    except Exception:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 64)).astype(np.float32)
        y = rng.integers(0, 10, size=(500,)).astype(int)
        return X, y, "synthetic_offline"


def _augment(images: np.ndarray, config: str, *, rng: np.random.Generator) -> np.ndarray:
    # images: (n, 8, 8)
    if config == "none":
        return images
    if config == "noise":
        noise = rng.normal(loc=0.0, scale=0.1, size=images.shape).astype(np.float32)
        return np.clip(images + noise, 0.0, 16.0)
    if config == "rotation":
        try:
            from scipy.ndimage import rotate  # type: ignore
            out = []
            for img in images:
                angle = float(rng.uniform(-15.0, 15.0))
                out.append(rotate(img, angle=angle, reshape=False, order=1, mode="nearest"))
            return np.stack(out).astype(np.float32)
        except Exception:
            return images
    if config == "flip":
        return np.flip(images, axis=2)
    if config == "combo":
        # B + C + D
        aug = _augment(images, "noise", rng=rng)
        aug = _augment(aug, "rotation", rng=rng)
        aug = _augment(aug, "flip", rng=rng)
        return aug
    return images


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    X, y, dataset_name = _load_digits()  # X: (n,64)
    n = X.shape[0]
    # Train/test split
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.metrics import accuracy_score  # type: ignore

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    Xtr_img = X_train.reshape(-1, 8, 8).astype(np.float32)
    Xte_flat = X_test.reshape(-1, 64).astype(np.float32)

    rng = np.random.default_rng(42)

    configs = [
        ("none", "none"),
        ("noise", "noise"),
        ("rotation", "rotation"),
        ("flip", "flip"),
        ("combo", "combo"),
    ]

    bars: List[float] = []
    bar_labels: List[str] = []
    comparison_table: List[Dict[str, Any]] = []
    best_acc: float = -1.0
    best_cfg: str | None = None

    try:
        for step_idx, (cfg_label, cfg_key) in enumerate(configs, start=1):
            _timeout_check()
            if cfg_key == "combo":
                aug_img = _augment(Xtr_img, "combo", rng=rng)
                X_aug = np.concatenate([Xtr_img, aug_img], axis=0)
            else:
                X_aug = _augment(Xtr_img, cfg_key, rng=rng)

            X_aug_flat = X_aug.reshape(X_aug.shape[0], -1)
            train_size_after_aug = int(X_aug_flat.shape[0])

            model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50, random_state=42, early_stopping=True)
            tfit0 = time.perf_counter()
            model.fit(X_aug_flat, y_train if cfg_key != "combo" else np.concatenate([y_train, y_train], axis=0))
            _timeout_check()
            preds = model.predict(Xte_flat)
            test_acc = float(accuracy_score(y_test, preds))
            emit("test_accuracy", test_acc, step_idx, model=cfg_label)
            emit("train_size_after_aug", float(train_size_after_aug), step_idx, model=cfg_label)

            bars.append(test_acc)
            bar_labels.append(cfg_label)
            comparison_table.append({"config": cfg_label, "test_accuracy": test_acc, "train_size_after_aug": train_size_after_aug})
            if test_acc > best_acc:
                best_acc = test_acc
                best_cfg = cfg_label

        # Plot bar chart.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        x = np.arange(len(bar_labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(bar_labels)))
        plt.bar(x, bars, color=colors, alpha=0.9)
        plt.xticks(x, bar_labels, rotation=20, ha="right", fontsize=9)
        plt.ylabel("Test Accuracy")
        plt.title(f"Augmentation Ablation ({dataset_name})")
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "computer_vision",
            "dataset": dataset_name,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_config": best_cfg, "best_test_accuracy": float(best_acc)},
            "comparison_table": comparison_table,
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(a)) for a in bars] if bars else [1.0],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "best_config": best_cfg, "acc": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "computer_vision",
            "dataset": dataset_name,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_config": best_cfg, "best_test_accuracy": float(best_acc)},
            "comparison_table": comparison_table,
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [best_acc, best_acc])
            plt.title("Timeout Augmentation Proxy")
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
            "domain": "computer_vision",
            "dataset": dataset_name,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": comparison_table,
            "acc": float(best_acc),
            "final_loss": float(1.0 - float(best_acc)),
            "losses": [float(1.0 - float(best_acc))],
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

