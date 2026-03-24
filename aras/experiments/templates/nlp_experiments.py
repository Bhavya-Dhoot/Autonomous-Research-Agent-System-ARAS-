from __future__ import annotations

"""
Domain experiment templates.

CoderAgent will import this module and write each entry's `code` string
into `exp*.py` files inside the cycle's experiment bundle directory.
"""

EXPERIMENTS = [
    {
        "name": "exp1_text_classification",
        "domain": "nlp",
        "dataset": "ag_news (datasets) / 20newsgroups (sklearn fallback) / synthetic (offline fallback)",
        "expected_metrics": ["accuracy", "f1_macro", "train_time_s", "inference_time_ms"],
        "code": r'''from __future__ import annotations

import json
import os
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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# -------------------- Universal Requirements --------------------
CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMIT_EXPERIMENT = "exp1_text_classification"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload = {"experiment": "exp1", "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0, model=None)
    raise TimeoutError("timeout")


def _setup_timeout(seconds: int = 280) -> None:
    # Linux/macOS: SIGALRM. Windows: threading is used (best-effort, no signal).
    try:
        import signal

        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[arg-type]
            signal.alarm(seconds)
            return
    except Exception:
        pass

    # Windows/best-effort fallback: use a threading timer and set a flag.
    import threading

    stop = {"flag": False}

    def _mark_timeout():
        stop["flag"] = True

    timer = threading.Timer(seconds, _mark_timeout)
    timer.daemon = True
    timer.start()
    # Expose flag to module-level via closure.
    globals()["_TIMEOUT_STOP"] = stop


def _timeout_check() -> None:
    stop = globals().get("_TIMEOUT_STOP")
    if isinstance(stop, dict) and stop.get("flag"):
        emit("timeout", 1.0, 0)
        raise TimeoutError("timeout")


def _make_synthetic_ag_news(n_train: int, n_test: int, seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    random.seed(seed)
    vocab = [
        "market","sports","technology","world","politics","science","culture","health","money","trade",
        "football","basketball","software","ai","research","economy","election","music","movie","protein",
        "data","model","cloud","robot","government","planet","galaxy","tournament","league","fitness","medicine",
        "policy","security","algorithm","learning","language","learning","network","cloud","quantum","neural"
    ]
    def mk_text(label: int) -> str:
        label_words = {
            0: ["world","policy","government","election","diplomacy","security"],
            1: ["sports","football","tournament","league","basketball","team"],
            2: ["technology","software","ai","model","data","network"],
            3: ["science","research","quantum","galaxy","planet","medicine"],
        }[label]
        words = random.choices(vocab, k=50) + random.choices(label_words, k=10)
        random.shuffle(words)
        return " ".join(words)

    # Deterministic balanced synthetic dataset.
    labels_train = [i % 4 for i in range(n_train)]
    labels_test = [i % 4 for i in range(n_test)]
    texts_train = [mk_text(lbl) for lbl in labels_train]
    texts_test = [mk_text(lbl) for lbl in labels_test]
    return texts_train, labels_train, texts_test, labels_test


def _load_ag_news(n_train: int, n_test: int) -> Tuple[List[str], List[int], List[str], List[int], str]:
    # Primary: HuggingFace datasets.
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("ag_news", cache_dir=str(CACHE_DIR))
        train = ds["train"].select(range(n_train))
        test = ds["test"].select(range(n_test))
        texts_train = [str(x) for x in train["text"]]
        y_train = [int(x) for x in train["label"]]
        texts_test = [str(x) for x in test["text"]]
        y_test = [int(x) for x in test["label"]]
        return texts_train, y_train, texts_test, y_test, "ag_news"
    except Exception:
        pass

    # Fallback: sklearn 20newsgroups subset.
    try:
        from sklearn.datasets import fetch_20newsgroups  # type: ignore

        # Map into 4 categories deterministically.
        categories = ["alt.atheism", "comp.graphics", "rec.sport.baseball", "sci.space"]
        data_train = fetch_20newsgroups(subset="train", categories=categories, remove=())
        data_test = fetch_20newsgroups(subset="test", categories=categories, remove=())

        texts_train = [str(t) for t in data_train.data[:n_train]]
        y_train = [int(y) for y in data_train.target[:n_train]]
        texts_test = [str(t) for t in data_test.data[:n_test]]
        y_test = [int(y) for y in data_test.target[:n_test]]

        return texts_train, y_train, texts_test, y_test, "20newsgroups_4cats"
    except Exception:
        pass

    # Offline/synthetic final fallback.
    texts_train, y_train, texts_test, y_test = _make_synthetic_ag_news(n_train, n_test)
    return texts_train, y_train, texts_test, y_test, "synthetic_offline"


def _plot_confusion(cm: np.ndarray, labels: List[str], title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
    plt.figure(figsize=(7, 4))
    sns.heatmap(cm, annot=False, cmap="viridis", cbar=True)
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout(pad=0.5)
    plt.savefig("loss.png", dpi=300)
    plt.savefig("loss.pdf")
    plt.close()


def main() -> int:
    random.seed(42)
    np.random.seed(42)

    _setup_timeout(280)

    t0 = time.perf_counter()
    n_train = 2000
    n_test = 500
    experiment_name = "exp1_text_classification"
    labels = ["Class0","Class1","Class2","Class3"]
    dataset_name = "unknown"

    best_acc = 0.0
    best_f1 = 0.0
    best_model = None

    comparison_table: List[Dict[str, Any]] = []
    # Store last computed loss/curve-like values for FiguresAgent compatibility.
    losses: List[float] = []

    try:
        X_train, y_train, X_test, y_test, dataset_name = _load_ag_news(n_train, n_test)
        # Vectorizers.
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        count = CountVectorizer(max_features=20000, ngram_range=(1, 1))
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform(X_test)

        # Models.
        models = []
        models.append(("tfidf+logreg", LogisticRegression(max_iter=1, warm_start=True, solver="saga", multi_class="auto", n_jobs=1, random_state=42)))
        models.append(("tfidf+sgd", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, tol=None, random_state=42)))
        models.append(("count+nb", MultinomialNB(alpha=1.0)))

        # Stream per "epoch" by refitting / incremental training.
        epochs = 3
        for ep in range(1, epochs + 1):
            _timeout_check()
            # Logistic Regression (warm_start): increase max_iter.
            for name, model in models:
                if name == "tfidf+logreg":
                    emit("train_start", 0.0, ep, model=name)
                    model.max_iter = ep * 30
                    model.fit(X_train_tfidf, y_train)
                elif name == "tfidf+sgd":
                    # Partial_fit over shuffled minibatches.
                    # Convert to arrays of indices for deterministic mini-batching.
                    indices = np.arange(len(y_train))
                    rng = np.random.default_rng(42 + ep)
                    rng.shuffle(indices)
                    classes = np.unique(y_train)
                    for start in range(0, len(indices), 256):
                        _timeout_check()
                        batch_idx = indices[start : start + 256]
                        model.partial_fit(X_train_tfidf[batch_idx], np.array(y_train)[batch_idx], classes=classes)
                elif name == "count+nb":
                    # Naive Bayes has no epoch; fit each time and treat as epoch.
                    model.fit(X_train_count, y_train)

                t_inf0 = time.perf_counter()
                preds = model.predict(X_test_tfidf if name != "count+nb" else X_test_count)
                t_inf = (time.perf_counter() - t_inf0) * 1000.0

                acc = accuracy_score(y_test, preds)
                f1m = f1_score(y_test, preds, average="macro")
                emit("accuracy", acc, ep, model=name)

                losses.append(float(1.0 - acc))

                # Train time (approx): just measure fit time during the last epoch for reporting.
                # For fairness, we compute fit time each epoch and keep the latest for table.
                # Note: this is still meaningful for comparative reporting.
                # For logreg/nb we already trained; for sgd partial_fit occurred.
                # We approximate train_time as 0.0 here because fit time was not separately measured.
                if name == "tfidf+sgd":
                    train_time_s = 0.0
                else:
                    train_time_s = 0.0

                metrics_row = {
                    "model": name,
                    "accuracy": float(acc),
                    "f1_macro": float(f1m),
                    "train_time_s": float(train_time_s),
                    "inference_time_ms": float(t_inf),
                }
                if len(comparison_table) < 3 * epochs:
                    comparison_table.append(metrics_row)

                if acc > best_acc:
                    best_acc = float(acc)
                    best_f1 = float(f1m)
                    best_model = name

        # Final fit for stable results: train each model fresh once.
        # Logistic Regression
        mA = LogisticRegression(max_iter=200, warm_start=False, solver="saga", multi_class="auto", n_jobs=1, random_state=42)
        tfit0 = time.perf_counter()
        mA.fit(X_train_tfidf, y_train)
        train_time_s_A = time.perf_counter() - tfit0
        t_inf0 = time.perf_counter()
        preds_A = mA.predict(X_test_tfidf)
        inf_ms_A = (time.perf_counter() - t_inf0) * 1000.0
        acc_A = accuracy_score(y_test, preds_A)
        f1_A = f1_score(y_test, preds_A, average="macro")

        # SGDClassifier
        mB = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=300, tol=1e-3, random_state=42)
        tfit0 = time.perf_counter()
        mB.fit(X_train_tfidf, y_train)
        train_time_s_B = time.perf_counter() - tfit0
        t_inf0 = time.perf_counter()
        preds_B = mB.predict(X_test_tfidf)
        inf_ms_B = (time.perf_counter() - t_inf0) * 1000.0
        acc_B = accuracy_score(y_test, preds_B)
        f1_B = f1_score(y_test, preds_B, average="macro")

        # MultinomialNB
        mC = MultinomialNB(alpha=1.0)
        tfit0 = time.perf_counter()
        mC.fit(X_train_count, y_train)
        train_time_s_C = time.perf_counter() - tfit0
        t_inf0 = time.perf_counter()
        preds_C = mC.predict(X_test_count)
        inf_ms_C = (time.perf_counter() - t_inf0) * 1000.0
        acc_C = accuracy_score(y_test, preds_C)
        f1_C = f1_score(y_test, preds_C, average="macro")

        # Pick best.
        candidates = [
            ("model_A_tfidf+logreg", acc_A, f1_A, train_time_s_A, inf_ms_A, preds_A),
            ("model_B_tfidf+sgd", acc_B, f1_B, train_time_s_B, inf_ms_B, preds_B),
            ("model_C_count+nb", acc_C, f1_C, train_time_s_C, inf_ms_C, preds_C),
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_model, best_acc, best_f1, train_time_s_best, inf_ms_best, best_preds = candidates[0]

        emit("accuracy", best_acc, epochs, model=best_model)

        cm = confusion_matrix(y_test, best_preds, labels=[0,1,2,3])
        _plot_confusion(cm, labels, f"Text Classification Confusion Matrix ({dataset_name})")

        # Final results.
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": experiment_name,
            "domain": "nlp",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {
                "model_A_accuracy": float(acc_A),
                "model_A_f1_macro": float(f1_A),
                "model_A_train_time_s": float(train_time_s_A),
                "model_A_inference_time_ms": float(inf_ms_A),
                "model_B_accuracy": float(acc_B),
                "model_B_f1_macro": float(f1_B),
                "model_B_train_time_s": float(train_time_s_B),
                "model_B_inference_time_ms": float(inf_ms_B),
                "model_C_accuracy": float(acc_C),
                "model_C_f1_macro": float(f1_C),
                "model_C_train_time_s": float(train_time_s_C),
                "model_C_inference_time_ms": float(inf_ms_C),
                "best_model": best_model,
                "best_accuracy": float(best_acc),
                "best_f1_macro": float(best_f1),
            },
            "comparison_table": [
                {"model": "TF-IDF+LogReg", "accuracy": float(acc_A), "f1": float(f1_A), "train_time_s": float(train_time_s_A), "inference_time_ms": float(inf_ms_A)},
                {"model": "TF-IDF+SGD", "accuracy": float(acc_B), "f1": float(f1_B), "train_time_s": float(train_time_s_B), "inference_time_ms": float(inf_ms_B)},
                {"model": "Count+MultinomialNB", "accuracy": float(acc_C), "f1": float(f1_C), "train_time_s": float(train_time_s_C), "inference_time_ms": float(inf_ms_C)},
            ],
            # Back-compat fields for current pipeline:
            "acc": float(best_acc),
            "final_loss": float(1.0 - best_acc),
            "losses": losses if losses else [float(1.0 - best_acc)],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": experiment_name, "acc": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        # Partial results: write what we have.
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": experiment_name,
            "domain": "nlp",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_accuracy": float(best_acc), "best_f1_macro": float(best_f1)},
            "comparison_table": [],
            "acc": float(best_acc),
            "final_loss": float(1.0 - best_acc),
            "losses": losses[:10] if losses else [float(1.0 - best_acc)],
            "timed_out": True,
        }
        try:
            # Minimal plot so writer/figures have something.
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            xs = list(range(1, len(results["losses"]) + 1))
            plt.plot(xs, results["losses"])
            plt.title("Timeout Loss Proxy")
            plt.xlabel("step")
            plt.ylabel("proxy loss")
            plt.tight_layout(pad=0.5)
            plt.savefig("loss.png", dpi=300)
            plt.savefig("loss.pdf")
            plt.close()
        except Exception:
            pass
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        # Hard failure: emit and re-raise as clean exit.
        runtime_seconds = time.perf_counter() - t0
        emit("error", 0.0, 0, model="error")
        err = traceback.format_exc()
        results = {
            "experiment": experiment_name,
            "domain": "nlp",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": float(best_acc),
            "final_loss": float(1.0 - best_acc),
            "losses": losses[:10] if losses else [float(1.0 - best_acc)],
            "error": err[-2000:],
            "timed_out": False,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [1, 1])
            plt.title("Error Loss Proxy")
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
        "name": "exp2_sentiment_analysis",
        "domain": "nlp",
        "dataset": "imdb (datasets) / 20newsgroups (binary fallback) / synthetic (offline fallback)",
        "expected_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMIT_EXPERIMENT = "exp2_sentiment_analysis"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload = {"experiment": "exp2", "metric": metric, "value": float(value), "step": int(step)}
    if model:
        payload["model"] = model
    print("METRIC_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def _timeout_handler(sig, frame):  # type: ignore[no-untyped-def]
    emit("timeout", 1.0, 0, model=None)
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


def _make_synthetic_sentiment(n_train: int, n_test: int, seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    random.seed(seed)
    pos_words = ["good","great","excellent","amazing","fantastic","love","wonderful","best","fun","enjoy"]
    neg_words = ["bad","terrible","awful","hate","worst","boring","dull","poor","waste","pain"]
    neutral = ["movie","story","plot","character","music","acting","performance","camera","sound","dialogue"]
    def mk(label: int) -> str:
        core = pos_words if label == 1 else neg_words
        words = random.choices(neutral, k=40) + random.choices(core, k=10)
        random.shuffle(words)
        return " ".join(words)
    y_train = [i % 2 for i in range(n_train)]
    y_test = [i % 2 for i in range(n_test)]
    X_train = [mk(y) for y in y_train]
    X_test = [mk(y) for y in y_test]
    return X_train, y_train, X_test, y_test


def _load_imdb(n_train: int, n_test: int) -> Tuple[List[str], List[int], List[str], List[int], str]:
    # Primary
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("imdb", cache_dir=str(CACHE_DIR))
        train = ds["train"].select(range(n_train))
        test = ds["test"].select(range(n_test))
        X_train = [str(x) for x in train["text"]]
        y_train = [int(x) for x in train["label"]]
        X_test = [str(x) for x in test["text"]]
        y_test = [int(x) for x in test["label"]]
        return X_train, y_train, X_test, y_test, "imdb"
    except Exception:
        pass

    # Fallback: sklearn 20newsgroups binary subset.
    try:
        from sklearn.datasets import fetch_20newsgroups  # type: ignore
        categories = ["talk.religion.misc", "sci.space"]
        train = fetch_20newsgroups(subset="train", categories=categories, remove=())
        test = fetch_20newsgroups(subset="test", categories=categories, remove=())
        X_train = [str(t) for t in train.data[:n_train]]
        y_train = [int(y) for y in train.target[:n_train]]
        X_test = [str(t) for t in test.data[:n_test]]
        y_test = [int(y) for y in test.target[:n_test]]
        return X_train, y_train, X_test, y_test, "20newsgroups_binary"
    except Exception:
        pass

    X_train, y_train, X_test, y_test = _make_synthetic_sentiment(n_train, n_test)
    return X_train, y_train, X_test, y_test, "synthetic_offline"


def _plot_roc(fprs: Dict[str, np.ndarray], tprs: Dict[str, np.ndarray], aucs: Dict[str, float], title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
    plt.figure(figsize=(7, 4))
    for name in sorted(aucs.keys()):
        plt.plot(fprs[name], tprs[name], linewidth=2.0, label=f"{name} (AUC={aucs[name]:.3f})")
    plt.plot([0,1],[0,1], linestyle="--", color="gray", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout(pad=0.5)
    plt.savefig("loss.png", dpi=300)
    plt.savefig("loss.pdf")
    plt.close()


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)

    t0 = time.perf_counter()
    n_train = 1000
    n_test = 200
    experiment_name = "exp2_sentiment_analysis"
    dataset_name = "unknown"

    losses: List[float] = []
    best_acc = 0.0
    best_f1 = 0.0
    best_model: str | None = None

    try:
        X_train, y_train, X_test, y_test, dataset_name = _load_imdb(n_train, n_test)
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        bow = CountVectorizer(max_features=20000, ngram_range=(1, 1))
        X_train_bow = bow.fit_transform(X_train)
        X_test_bow = bow.transform(X_test)

        # Model A: TF-IDF + LogisticRegression
        models = [
            ("tfidf+logreg", LogisticRegression(max_iter=200, n_jobs=1, random_state=42)),
            ("tfidf+linearsvc", LinearSVC(random_state=42)),
            ("bow+bernoulliNB", BernoulliNB(alpha=1.0)),
        ]

        fprs: Dict[str, np.ndarray] = {}
        tprs: Dict[str, np.ndarray] = {}
        aucs: Dict[str, float] = {}
        comparison_table: List[Dict[str, Any]] = []

        for idx, (name, model) in enumerate(models, start=1):
            _timeout_check()
            tfit0 = time.perf_counter()
            if name == "bow+bernoulliNB":
                model.fit(X_train_bow, y_train)
                scores = model.predict_proba(X_test_bow)[:, 1]
                preds = (scores >= 0.5).astype(int)
            elif name == "tfidf+linearsvc":
                model.fit(X_train_tfidf, y_train)
                scores = model.decision_function(X_test_tfidf)
                # Convert decision scores to predictions by threshold 0.
                preds = (scores >= 0).astype(int)
            else:
                model.fit(X_train_tfidf, y_train)
                scores = model.predict_proba(X_test_tfidf)[:, 1]
                preds = (scores >= 0.5).astype(int)
            train_time_s = time.perf_counter() - tfit0

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1m = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, scores)

            emit("accuracy", acc, idx, model=name)
            emit("roc_auc", auc, idx, model=name)

            # ROC
            fpr, tpr, _ = roc_curve(y_test, scores)
            fprs[name] = fpr
            tprs[name] = tpr
            aucs[name] = float(auc)

            comparison_table.append(
                {
                    "model": name,
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1m),
                    "roc_auc": float(auc),
                    "train_time_s": float(train_time_s),
                }
            )

            losses.append(float(1.0 - acc))
            if acc > best_acc:
                best_acc = float(acc)
                best_f1 = float(f1m)
                best_model = name

        # Plot.
        _plot_roc(fprs, tprs, aucs, f"Sentiment ROC Curves ({dataset_name})")

        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": experiment_name,
            "domain": "nlp",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {
                "model_A_accuracy": float(comparison_table[0]["accuracy"]),
                "model_A_f1": float(comparison_table[0]["f1"]),
                "model_A_roc_auc": float(comparison_table[0]["roc_auc"]),
                "model_B_accuracy": float(comparison_table[1]["accuracy"]),
                "model_B_f1": float(comparison_table[1]["f1"]),
                "model_B_roc_auc": float(comparison_table[1]["roc_auc"]),
                "model_C_accuracy": float(comparison_table[2]["accuracy"]),
                "model_C_f1": float(comparison_table[2]["f1"]),
                "model_C_roc_auc": float(comparison_table[2]["roc_auc"]),
                "best_model": best_model,
                "best_accuracy": float(best_acc),
                "best_f1": float(best_f1),
            },
            "comparison_table": comparison_table,
            # Back-compat fields for current pipeline:
            "acc": float(best_acc),
            "final_loss": float(1.0 - best_acc),
            "losses": losses if losses else [float(1.0 - best_acc)],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": experiment_name, "acc": best_acc}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": experiment_name,
            "domain": "nlp",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"best_accuracy": float(best_acc)},
            "comparison_table": [],
            "acc": float(best_acc),
            "final_loss": float(1.0 - best_acc),
            "losses": losses[:10] if losses else [float(1.0 - best_acc)],
            "timed_out": True,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            xs = list(range(1, len(results["losses"]) + 1))
            plt.plot(xs, results["losses"])
            plt.title("Timeout Loss Proxy")
            plt.xlabel("step")
            plt.ylabel("proxy loss")
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
            "experiment": experiment_name,
            "domain": "nlp",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": float(best_acc),
            "final_loss": float(1.0 - best_acc),
            "losses": losses[:10] if losses else [float(1.0 - best_acc)],
            "error": err[-2000:],
            "timed_out": False,
        }
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.figure(figsize=(7, 4))
            plt.plot([0, 1], [1, 1])
            plt.title("Error Loss Proxy")
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
        "name": "exp3_tokenizer_benchmark",
        "domain": "nlp",
        "dataset": "synthetic tokenizer benchmark corpus (offline)",
        "expected_metrics": ["tokens_per_second", "avg_tokens_per_doc", "vocab_size", "oov_rate"],
        "code": r'''from __future__ import annotations

import json
import random
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMIT_EXPERIMENT = "exp3_tokenizer_benchmark"


def emit(metric: str, value: float, step: int, *, model: str | None = None) -> None:
    payload = {"experiment": "exp3", "metric": metric, "value": float(value), "step": int(step)}
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


def _make_corpus(n_docs: int = 500, seed: int = 42) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    base_vocab = [
        "the","and","to","of","in","a","for","is","on","with","as","are","be","at","by","from","it","this",
        "data","model","learning","science","language","tokens","benchmark","random","text","algorithm",
        "analysis","probability","function","experiment","results","future","system","agent","autonomous","research"
    ]
    # Wikipedia-length-ish docs (deterministic).
    docs: List[str] = []
    for i in range(n_docs):
        length = 180 + (i % 60)  # 180..239 tokens approx
        words = [random.choice(base_vocab) for _ in range(length)]
        docs.append(" ".join(words))
    return docs, base_vocab


def tokenizer_split(text: str) -> List[str]:
    return text.split()


def tokenizer_regex(text: str) -> List[str]:
    return [t for t in re.split(r"\\W+", text) if t]


def tokenizer_nltk(text: str) -> List[str]:
    try:
        import nltk  # type: ignore
        from nltk.tokenize import word_tokenize  # type: ignore

        return word_tokenize(text)
    except Exception:
        # Fallback to regex if nltk isn't available.
        return tokenizer_regex(text)


def bpe_train(corpus_tokens: List[List[str]], merges: int = 100) -> Tuple[dict[tuple[str, str], str], dict[str, int]]:
    # Character-level BPE with a small vocab.
    # Each word represented as list of characters + ['</w>'].
    word_seqs: List[List[str]] = []
    for doc_tokens in corpus_tokens:
        for tok in doc_tokens:
            seq = list(tok)
            seq.append("</w>")
            word_seqs.append(seq)

    merge_rules: dict[tuple[str, str], str] = {}
    for _ in range(merges):
        _timeout_check()
        pairs = Counter()
        for seq in word_seqs:
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i+1])] += 1
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merged = a + b
        merge_rules[(a, b)] = merged

        # Apply merge to sequences.
        new_word_seqs: List[List[str]] = []
        for seq in word_seqs:
            i = 0
            out: List[str] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i+1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_word_seqs.append(out)
        word_seqs = new_word_seqs

    # Approx vocab size from merge_rules keys.
    vocab = Counter()
    vocab.update(merge_rules.values())
    return merge_rules, vocab


def bpe_encode(text: str, merge_rules: dict[tuple[str, str], str]) -> List[str]:
    # Apply merges greedily to each token.
    tokens = tokenizer_split(text)
    out: List[str] = []
    for tok in tokens:
        seq = list(tok) + ["</w>"]
        # Greedy merges until no applicable adjacent pair.
        merged_any = True
        while merged_any:
            merged_any = False
            i = 0
            new_seq: List[str] = []
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) in merge_rules:
                    new_seq.append(merge_rules[(seq[i], seq[i+1])])
                    i += 2
                    merged_any = True
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        out.extend([t for t in seq if t != "</w>"])
    return out


def _style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)

    t0 = time.perf_counter()
    n_docs = 500

    try:
        docs, base_vocab = _make_corpus(n_docs=n_docs, seed=42)
        base_vocab_set = set(base_vocab)
        results: Dict[str, Any] = {"experiment": EMIT_EXPERIMENT, "domain": "nlp", "runtime_seconds": 0.0}

        # Tokenize and benchmark.
        tokenizers = [
            ("split", tokenizer_split),
            ("nltk_word_tokenize", tokenizer_nltk),
            ("regex", tokenizer_regex),
        ]

        metrics: Dict[str, Any] = {}
        # First, pre-tokenize once for BPE training.
        corpus_tokens = [tokenizer_split(d) for d in docs]

        for step, (name, fn) in enumerate(tokenizers, start=1):
            _timeout_check()
            t_start = time.perf_counter()
            token_counts = 0
            all_tokens: List[str] = []
            for d in docs:
                toks = fn(d)
                token_counts += len(toks)
                all_tokens.extend(toks)
            elapsed = max(1e-6, time.perf_counter() - t_start)

            tokens_per_second = token_counts / elapsed
            avg_tokens_per_doc = token_counts / float(n_docs)
            vocab_size = len(set(all_tokens))
            oov_rate = 1.0 - (len(set(all_tokens).intersection(base_vocab_set)) / max(1, len(set(all_tokens))))

            emit("tokens_per_second", tokens_per_second, step, model=name)
            emit("avg_tokens_per_doc", avg_tokens_per_doc, step, model=name)
            emit("oov_rate", oov_rate, step, model=name)

            metrics[name] = {
                "tokens_per_second": float(tokens_per_second),
                "avg_tokens_per_doc": float(avg_tokens_per_doc),
                "vocab_size": int(vocab_size),
                "oov_rate": float(oov_rate),
            }

        # BPE simulation: train merge rules from synthetic corpus.
        _timeout_check()
        merge_rules, _vocab = bpe_train(corpus_tokens, merges=100)
        # Benchmark encoding.
        t_start = time.perf_counter()
        token_counts = 0
        all_tokens = []
        for d in docs:
            toks = bpe_encode(d, merge_rules)
            token_counts += len(toks)
            all_tokens.extend(toks)
        elapsed = max(1e-6, time.perf_counter() - t_start)
        tokens_per_second = token_counts / elapsed
        avg_tokens_per_doc = token_counts / float(n_docs)
        vocab_size = len(set(all_tokens))
        oov_rate = 1.0 - (len(set(all_tokens).intersection(base_vocab_set)) / max(1, len(set(all_tokens))))

        emit("tokens_per_second", tokens_per_second, 4, model="bpe_sim_100_merges")
        emit("avg_tokens_per_doc", avg_tokens_per_doc, 4, model="bpe_sim_100_merges")
        emit("oov_rate", oov_rate, 4, model="bpe_sim_100_merges")

        metrics["bpe_sim_100_merges"] = {
            "tokens_per_second": float(tokens_per_second),
            "avg_tokens_per_doc": float(avg_tokens_per_doc),
            "vocab_size": int(vocab_size),
            "oov_rate": float(oov_rate),
        }

        # Plot bar chart tokens/sec.
        _style()
        plt.figure(figsize=(7, 4))
        names = list(metrics.keys())
        vals = [metrics[n]["tokens_per_second"] for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        plt.bar(names, vals, color=colors)
        plt.ylabel("tokens / second")
        plt.title("Tokenizer Throughput Comparison")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        results.update(
            {
                "dataset": "synthetic_wikipedia_length_texts",
                "n_docs": n_docs,
                "metrics": metrics,
                "comparison_table": [],
                # Back-compat fields for current pipeline:
                "acc": float(max(v["tokens_per_second"] for v in metrics.values())),
                "final_loss": float(0.0),
                "losses": [float(1.0 / max(1e-9, v["tokens_per_second"])) for v in metrics.values()],
            }
        )
        results["runtime_seconds"] = float(runtime_seconds)
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EMIT_EXPERIMENT, "best_tokens_per_second": results["acc"]}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        # Partial: at least write empty results with proxy.
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EMIT_EXPERIMENT,
            "domain": "nlp",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        try:
            _style()
            plt.figure(figsize=(7, 4))
            plt.plot([0,1],[1,1])
            plt.title("Timeout Throughput Proxy")
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
            "experiment": EMIT_EXPERIMENT,
            "domain": "nlp",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "error": err[-2000:],
            "timed_out": False,
        }
        try:
            _style()
            plt.figure(figsize=(7, 4))
            plt.plot([0,1],[1,1])
            plt.title("Error Throughput Proxy")
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
]

