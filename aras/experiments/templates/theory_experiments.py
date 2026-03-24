from __future__ import annotations

"""
Theory experiment templates.
"""

EXPERIMENTS = [
    {
        "name": "exp1_algorithm_complexity",
        "domain": "theory",
        "dataset": "synthetic arrays (seeded random)",
        "expected_metrics": ["mean_wall_time_s"],
        "code": r'''from __future__ import annotations

import json
import math
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
EXPERIMENT_NAME = "exp1_algorithm_complexity"


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


def bubble_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    out: List[int] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def quick_sort(arr: List[int]) -> List[int]:
    a = arr[:]
    if len(a) <= 1:
        return a
    pivot = a[random.randint(0, len(a) - 1)]
    left = [x for x in a if x < pivot]
    mid = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)


def heap_sort(arr: List[int]) -> List[int]:
    # Simple heap sort using Python heapq.
    import heapq
    h = arr[:]
    heapq.heapify(h)
    out = []
    while h:
        out.append(heapq.heappop(h))
    return out


def _fit_slope_loglog(ns: List[int], ts: List[float]) -> float:
    # slope of log(time) ~ slope*log(n) + c
    if len(ns) < 2:
        return 0.0
    ln = np.log(np.array(ns, dtype=np.float64))
    lt = np.log(np.array(ts, dtype=np.float64) + 1e-12)
    slope = float(np.polyfit(ln, lt, 1)[0])
    return slope


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    ns = [100, 500, 1000, 2500, 5000, 10000]
    trials = 5

    algos = [
        ("bubble_sort", bubble_sort),
        ("merge_sort", merge_sort),
        ("quick_sort", quick_sort),
        ("tim_sort", lambda a: sorted(a)),
        ("heap_sort", heap_sort),
    ]

    timings: Dict[str, Dict[int, List[float]]] = {name: {} for name, _ in algos}
    step_idx = 0

    try:
        for n in ns:
            for trial in range(trials):
                _timeout_check()
                arr = [int(x) for x in np.random.default_rng(42 + trial).integers(low=-100000, high=100000, size=n)]
                for algo_name, fn in algos:
                    _timeout_check()
                    tt0 = time.perf_counter()
                    _ = fn(arr)
                    dt = time.perf_counter() - tt0
                    timings[algo_name].setdefault(n, []).append(float(dt))
                    step_idx += 1
                    emit("mean_wall_time_s", float(dt), step_idx, model=f"{algo_name}_n{n}")

        # Aggregate means.
        means: Dict[str, List[float]] = {}
        slopes: Dict[str, float] = {}
        for algo_name, _ in algos:
            mean_ts = []
            ns_used = []
            for n in ns:
                vals = timings[algo_name].get(n) or []
                if not vals:
                    continue
                mean_ts.append(float(np.mean(vals)))
                ns_used.append(n)
            means[algo_name] = mean_ts
            slopes[algo_name] = _fit_slope_loglog(ns_used, mean_ts)

        # Plots: log-log time vs n.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        plt.figure(figsize=(7, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))
        for i, (algo_name, _) in enumerate(algos):
            ys = means.get(algo_name) or []
            if not ys:
                continue
            plt.plot(ns[:len(ys)], ys, marker="o", linewidth=2.0, label=f"{algo_name} (slope~{slopes.get(algo_name,0.0):.2f})", color=colors[i])
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("n (log scale)")
        plt.ylabel("wall time (s, log scale)")
        plt.title("Empirical Sorting Complexity")
        plt.legend(fontsize=8)
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        # Choose best by fastest at n=10000 if present.
        fastest = None
        fastest_time = 1e18
        for algo_name, _ in algos:
            vals = timings[algo_name].get(10000) or []
            if not vals:
                continue
            m = float(np.mean(vals))
            if m < fastest_time:
                fastest_time = m
                fastest = algo_name

        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "theory",
            "dataset": "synthetic_random_arrays",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"fastest_algo_at_10000": fastest, "fastest_wall_time_s": float(fastest_time) if fastest else 0.0},
            "comparison_table": [{"algo": name, "slope": float(slopes.get(name, 0.0))} for name, _ in algos],
            "acc": float(-fastest_time) if fastest else 0.0,
            "final_loss": float(0.0),
            "losses": [1.0],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "fastest": fastest, "slope": slopes.get(fastest or "", 0.0)}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        # Partial: plot whatever is available.
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
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "theory",
            "dataset": "synthetic_random_arrays",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {"timed_out": True},
            "comparison_table": [],
            "acc": 0.0,
            "final_loss": 0.0,
            "losses": [1.0],
            "timed_out": True,
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        runtime_seconds = time.perf_counter() - t0
        err = traceback.format_exc()
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "theory",
            "dataset": "synthetic_random_arrays",
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
    {
        "name": "exp2_numerical_bounds_verification",
        "domain": "theory",
        "dataset": "synthetic distributions (normal/exponential/coin)",
        "expected_metrics": ["chebyshev_empirical_prob", "clt_ks_statistic", "lln_running_mean_error"],
        "code": r'''from __future__ import annotations

import json
import math
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
EXPERIMENT_NAME = "exp2_numerical_bounds_verification"


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
        from scipy.stats import ks_1samp  # type: ignore
    except Exception:
        ks_1samp = None

    step_idx = 0
    results: Dict[str, Any] = {"experiment": EXPERIMENT_NAME, "domain": "theory", "runtime_seconds": 0.0}

    try:
        # A) Chebyshev inequality
        ks = [1, 2, 3, 4, 5]
        N = 20000
        cheb_emp: List[float] = []
        cheb_bound: List[float] = []
        for k in ks:
            _timeout_check()
            X = np.random.normal(loc=0.0, scale=1.0, size=(N,))
            mu = float(np.mean(X))
            sigma = float(np.std(X) + 1e-9)
            # For N(0,1), theoretical is 1/k^2.
            prob = float(np.mean(np.abs(X - mu) >= float(k) * sigma))
            bound = float(1.0 / (k * k))
            step_idx += 1
            emit("chebyshev_empirical_prob", prob, step_idx, model=f"k{k}")
            emit("chebyshev_theoretical_bound", bound, step_idx, model=f"k{k}")
            cheb_emp.append(prob)
            cheb_bound.append(bound)

        # B) CLT: KS statistic vs Normal
        n_list = [5, 10, 25, 100]
        lambda_rate = 1.0
        ks_stats: List[float] = []
        for n in n_list:
            _timeout_check()
            M = 10000
            samples = np.random.exponential(scale=1.0 / lambda_rate, size=(M, n)).mean(axis=1)
            # Normal with mean 1/lambda and variance 1/(n*lambda^2)
            mean = float(1.0 / lambda_rate)
            var = float(1.0 / (n * lambda_rate * lambda_rate))
            std = math.sqrt(var)
            if ks_1samp is not None:
                # Compute KS statistic vs normal CDF.
                import scipy.stats as st  # type: ignore
                d = float(st.kstest(samples, "norm", args=(mean, std)).statistic)
            else:
                # Fallback approximate KS: compare empirical vs normal CDF at quantiles.
                xs = np.sort(samples)
                cdf_emp = np.arange(1, len(xs) + 1) / len(xs)
                cdf_norm = 0.5 * (1 + np.erf((xs - mean) / (std * math.sqrt(2) + 1e-12)))
                d = float(np.max(np.abs(cdf_emp - cdf_norm)))
            step_idx += 1
            emit("clt_ks_statistic", d, step_idx, model=f"n{n}")
            ks_stats.append(d)

        # C) LLN: running mean
        p = 0.3
        flips = 10000
        running = []
        mean_val = 0.0
        err_running: List[float] = []
        for i in range(1, flips + 1):
            _timeout_check()
            x = 1.0 if np.random.random() < p else 0.0
            mean_val += (x - mean_val) / float(i)
            if i % 1000 == 0:
                running.append(float(mean_val))
                err_running.append(float(abs(mean_val - p)))
                step_idx += 1
                emit("lln_running_mean_error", err_running[-1], step_idx, model=f"n{i}")

        # Plot combined figure (3 parts).
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        # Chebyshev
        axs[0].plot(ks, cheb_emp, marker="o", label="Empirical P(|X|>=kσ)")
        axs[0].plot(ks, cheb_bound, marker="o", label="Theoretical 1/k^2")
        axs[0].set_title("Chebyshev Inequality")
        axs[0].set_xlabel("k")
        axs[0].set_ylabel("probability")
        axs[0].legend(fontsize=8)
        # CLT KS
        axs[1].plot(n_list, ks_stats, marker="o")
        axs[1].set_title("CLT: KS statistic vs Normal")
        axs[1].set_xlabel("n")
        axs[1].set_ylabel("KS")
        # LLN
        axs[2].plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000][: len(running)], running, marker="o")
        axs[2].axhline(p, linestyle="--", color="gray")
        axs[2].set_title("LLN: Running mean")
        axs[2].set_xlabel("n")
        axs[2].set_ylabel("running mean")
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        results.update(
            {
                "dataset": "normal_exponential_coin",
                "n_train": None,
                "n_test": None,
                "runtime_seconds": float(runtime_seconds),
                "metrics": {"chebyshev_emp": cheb_emp, "clt_ks": ks_stats, "lln_running_mean": running[-1] if running else 0.0},
                "comparison_table": [
                    {"k": k, "empirical": float(e), "bound": float(b)} for k, e, b in zip(ks, cheb_emp, cheb_bound)
                ],
                "acc": float(1.0 - (running[-1] - p) ** 2) if running else 0.0,
                "final_loss": 0.0,
                "losses": [1.0],
            }
        )
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "lln_final_mean": running[-1] if running else None}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results.update(
            {
                "runtime_seconds": float(runtime_seconds),
                "metrics": {},
                "comparison_table": [],
                "acc": 0.0,
                "final_loss": 0.0,
                "losses": [1.0],
                "timed_out": True,
            }
        )
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
        results.update(
            {
                "runtime_seconds": float(runtime_seconds),
                "metrics": {},
                "comparison_table": [],
                "acc": 0.0,
                "final_loss": 0.0,
                "losses": [1.0],
                "error": err[-2000:],
                "timed_out": False,
            }
        )
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    },
    {
        "name": "exp3_information_theory",
        "domain": "theory",
        "dataset": "20newsgroups (fallback synthetic)",
        "expected_metrics": ["entropy_char", "mutual_information", "kl_divergence", "perplexity"],
        "code": r'''from __future__ import annotations

import json
import math
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
import seaborn as sns


CACHE_DIR = Path.home() / ".aras_datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_NAME = "exp3_information_theory"


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


def _tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.split(r"\\W+", text) if t]


def _make_synthetic(categories: List[str], n_docs_per_cat: int) -> Dict[str, List[str]]:
    # Offline fallback: category-specific keyword vocab.
    vocab = {
        categories[0]: ["game","team","season","score","league","player","match","tournament","win"],
        categories[1]: ["space","planet","orbit","galaxy","stars","nasa","mission","telescope","rocket"],
        categories[2]: ["religion","faith","church","belief","god","prayer","scripture","bible","worship"],
        categories[3]: ["computer","software","programming","algorithm","data","model","system","network","code"],
    }
    rng = np.random.default_rng(42)
    out: Dict[str, List[str]] = {}
    for cat in categories:
        docs: List[str] = []
        for i in range(n_docs_per_cat):
            words = rng.choice(vocab[cat], size=120, replace=True).tolist()
            docs.append(" ".join(words))
        out[cat] = docs
    return out


def _load_20newsgroups(categories: List[str], n_docs_per_cat: int) -> Dict[str, List[str]]:
    # Primary: sklearn fetch of 20newsgroups.
    try:
        from sklearn.datasets import fetch_20newsgroups  # type: ignore
        data = fetch_20newsgroups(subset="train", categories=categories, remove=(), download_if_missing=True)
        out: Dict[str, List[str]] = {c: [] for c in categories}
        # fetch_20newsgroups returns numeric targets mapping to categories.
        for text, target in zip(data.data, data.target):
            c = categories[int(target)]
            if len(out[c]) < n_docs_per_cat:
                out[c].append(str(text))
            if all(len(out[c]) >= n_docs_per_cat for c in categories):
                break
        return out
    except Exception:
        return _make_synthetic(categories, n_docs_per_cat)


def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for _, c in counts.items():
        p = c / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return float(ent)


def _kl_divergence(p: Dict[str, float], q: Dict[str, float], *, eps: float = 1e-12) -> float:
    # KL(P||Q) with smoothing eps.
    keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for k in keys:
        pk = float(p.get(k, 0.0)) + eps
        qk = float(q.get(k, 0.0)) + eps
        kl += pk * math.log(pk / qk, 2)
    return float(kl)


def main() -> int:
    random.seed(42)
    np.random.seed(42)
    _setup_timeout(280)
    t0 = time.perf_counter()

    cats = ["talk.religion.misc", "sci.space", "rec.sport.baseball", "comp.graphics"]
    n_docs_per_cat = 200
    step_idx = 0

    try:
        data_by_cat = _load_20newsgroups(cats, n_docs_per_cat)

        # Build top 500 words across all categories.
        all_tokens: List[str] = []
        for c in cats:
            for doc in data_by_cat.get(c, []):
                all_tokens.extend(_tokenize_words(doc))
        vocab = [w for w, _ in Counter(all_tokens).most_common(500)]
        vocab_set = set(vocab)

        # A) Character-level entropy per category.
        char_entropies: Dict[str, float] = {}
        for c in cats:
            _timeout_check()
            counts = Counter()
            for doc in data_by_cat.get(c, []):
                counts.update(list(doc))
            ent = _entropy_from_counts(counts)
            char_entropies[c] = float(ent)
            step_idx += 1
            emit("entropy_char", float(ent), step_idx, model=c)

        # B) Mutual information between word presence and category:
        # We'll average MI across the top vocab words.
        # For each word w, treat X=presence (0/1) and Y=category.
        def mi_for_word(w: str) -> float:
            # contingency counts over categories.
            total_docs = sum(len(data_by_cat[cat]) for cat in cats)
            if total_docs == 0:
                return 0.0
            # compute presence probability and category probability
            p_y = {cat: len(data_by_cat[cat]) / total_docs for cat in cats}
            # P(X=1|Y)
            # For each category: count docs containing w.
            present_counts = {cat: 0 for cat in cats}
            for cat in cats:
                for doc in data_by_cat.get(cat, []):
                    toks = set(_tokenize_words(doc))
                    if w in toks:
                        present_counts[cat] += 1
            # MI: sum_{x,y} p(x,y) log p(x,y)/(p(x)p(y))
            p_x = {}
            for x in [0, 1]:
                # aggregate across categories
                p_x[x] = 0.0
            for cat in cats:
                py = p_y[cat]
                p_x1y = present_counts[cat] / max(1, len(data_by_cat[cat]))
                p_x[1] += py * p_x1y
                p_x[0] += py * (1.0 - p_x1y)
            mi = 0.0
            for cat in cats:
                py = p_y[cat]
                p_x1y = present_counts[cat] / max(1, len(data_by_cat[cat]))
                p_x0y = 1.0 - p_x1y
                # x=1
                p_xy1 = py * p_x1y
                if p_xy1 > 0:
                    mi += p_xy1 * math.log(p_xy1 / max(1e-12, p_x[1] * py), 2)
                # x=0
                p_xy0 = py * p_x0y
                if p_xy0 > 0:
                    mi += p_xy0 * math.log(p_xy0 / max(1e-12, p_x[0] * py), 2)
            return float(mi)

        # compute MI for first 200 words to keep runtime reasonable
        mi_values: List[float] = []
        for i, w in enumerate(vocab[:200], start=1):
            _timeout_check()
            mi = mi_for_word(w)
            mi_values.append(float(mi))
            if i % 50 == 0:
                step_idx += 1
                emit("mutual_information_partial_avg", float(np.mean(mi_values)), step_idx, model="mi_top_words")

        mi_avg = float(np.mean(mi_values)) if mi_values else 0.0
        step_idx += 1
        emit("mutual_information", mi_avg, step_idx, model="avg_top_words")

        # C) KL divergence between pairs categories using unigram word distributions over vocab.
        def unigram_distribution(cat: str) -> Dict[str, float]:
            counts = Counter()
            for doc in data_by_cat.get(cat, []):
                toks = _tokenize_words(doc)
                counts.update([t for t in toks if t in vocab_set])
            total = sum(counts.values())
            if total <= 0:
                return {w: 1.0 / max(1, len(vocab_set)) for w in vocab[:50]}
            return {w: float(counts.get(w, 0)) / total for w in vocab}

        dists = {cat: unigram_distribution(cat) for cat in cats}
        kl_pairs: List[Tuple[str, str, float]] = []
        kl_heat = np.zeros((len(cats), len(cats)), dtype=np.float32)
        for i, pcat in enumerate(cats):
            for j, qcat in enumerate(cats):
                if i == j:
                    continue
                _timeout_check()
                kl = _kl_divergence(dists[pcat], dists[qcat])
                kl_heat[i, j] = float(kl)
                kl_pairs.append((pcat, qcat, float(kl)))
                step_idx += 1
                emit("kl_divergence", float(kl), step_idx, model=f"{pcat}->{qcat}")

        # D) Perplexity of unigram LM per category (compute cross-entropy on its own docs).
        def perplexity_for_cat(cat: str) -> float:
            # Train unigram LM on first half, evaluate on second half.
            docs = data_by_cat.get(cat, [])
            if not docs:
                return 0.0
            split = max(1, len(docs) // 2)
            train_docs = docs[:split]
            test_docs = docs[split:]
            counts = Counter()
            for doc in train_docs:
                toks = [t for t in _tokenize_words(doc) if t in vocab_set]
                counts.update(toks)
            total = sum(counts.values())
            # Smoothing
            denom = float(total + 1e-9)
            # Evaluate
            log_prob_sum = 0.0
            n_tokens = 0
            for doc in test_docs:
                toks = [t for t in _tokenize_words(doc) if t in vocab_set]
                for t in toks:
                    p = (counts.get(t, 0) + 1e-12) / denom
                    log_prob_sum += math.log(p + 1e-30)
                    n_tokens += 1
            if n_tokens == 0:
                return 0.0
            # Perplexity using natural logs: exp(-avg_log_prob)
            ppl = math.exp(-log_prob_sum / float(n_tokens))
            return float(ppl)

        perplexities: Dict[str, float] = {}
        for c in cats:
            _timeout_check()
            ppl = perplexity_for_cat(c)
            perplexities[c] = float(ppl)
            step_idx += 1
            emit("perplexity", float(ppl), step_idx, model=c)

        # Plot: 4 subplots + KL heatmap.
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs = axs.flatten()
        # Entropy bar
        axs[0].bar(cats, [char_entropies[c] for c in cats], color=plt.cm.Set2(np.linspace(0, 1, len(cats))))
        axs[0].set_title("Char entropy H(X)")
        axs[0].set_xticklabels(cats, rotation=20, ha="right")
        # MI single value
        axs[1].axis("off")
        axs[1].text(0.1, 0.5, f"Mutual information (avg top words)\\nI(X;Y)={mi_avg:.4f} bits", fontsize=12)
        # Perplexity bar
        axs[2].bar(cats, [perplexities[c] for c in cats], color=plt.cm.Set2(np.linspace(0, 1, len(cats))))
        axs[2].set_title("Perplexity (unigram LM)")
        axs[2].set_xticklabels(cats, rotation=20, ha="right")
        # KL heatmap
        sns.heatmap(kl_heat, ax=axs[3], cmap="viridis", cbar=True, xticklabels=cats, yticklabels=cats)
        axs[3].set_title("KL divergence P||Q (bits)")
        # Empty slots
        axs[4].axis("off")
        axs[5].axis("off")
        plt.tight_layout(pad=0.5)
        plt.savefig("loss.png", dpi=300)
        plt.savefig("loss.pdf")
        plt.close()

        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "theory",
            "dataset": "20newsgroups_or_synthetic",
            "runtime_seconds": float(runtime_seconds),
            "metrics": {
                "char_entropy": char_entropies,
                "mutual_information": mi_avg,
                "perplexity": perplexities,
            },
            "comparison_table": [{"pair": f"{a}->{b}", "kl_bits": v} for a, b, v in kl_pairs[:10]],
            "acc": float(mi_avg),
            "final_loss": 0.0,
            "losses": [1.0],
        }
        Path("results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"experiment": EXPERIMENT_NAME, "mi_avg_bits": mi_avg}, ensure_ascii=False), flush=True)
        return 0
    except TimeoutError:
        runtime_seconds = time.perf_counter() - t0
        results = {
            "experiment": EXPERIMENT_NAME,
            "domain": "theory",
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
            "domain": "theory",
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

