from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.experiments.runner import run_experiment_module
from aras.healing.fallback_router import FallbackRouter
from aras.self_improvement.failure_taxonomy import classify_failure
from aras.utils.fs import safe_write_text


@dataclass
class ExperimentBundle:
    slug: str
    root: Path
    modules: list[Path]
    tests: list[Path]


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "topic"


def _extract_code_placeholders(code: str, *, topic: str, plan: dict[str, Any], domain: str) -> str:
    # Optional topic injection if template uses placeholders.
    hypothesis = str(plan.get("hypothesis") or "")
    keywords = plan.get("keywords") or []
    kw_text = ", ".join([str(x) for x in keywords])
    out = code
    out = out.replace("{{TOPIC}}", topic)
    out = out.replace("{{HYPOTHESIS}}", hypothesis)
    out = out.replace("{{KEYWORDS}}", kw_text)
    out = out.replace("{{DOMAIN}}", domain)
    return out


def _classify_domain(*, plan: dict[str, Any], topic: str) -> str:
    # Domain keywords mapping from the spec.
    domain_keywords: dict[str, list[str]] = {
        "nlp": [
            "nlp",
            "text",
            "language",
            "sentiment",
            "classification",
            "tokeniz",
            "corpus",
            "embedding",
            "transformer",
            "bert",
            "gpt",
            "translation",
            "summarization",
            "qa",
            "question",
        ],
        "computer_vision": [
            "vision",
            "image",
            "visual",
            "cnn",
            "convolution",
            "object detection",
            "segmentation",
            "pixel",
            "recognition",
            "mnist",
            "cifar",
            "resnet",
        ],
        "reinforcement_learning": [
            "reinforcement",
            "rl",
            "reward",
            "policy",
            "agent",
            "environment",
            "bandit",
            "q-learning",
            "markov",
            "mdp",
            "gym",
            "cartpole",
        ],
        "theory": [
            "theorem",
            "proof",
            "complexity",
            "bound",
            "convergence",
            "information theory",
            "entropy",
            "algorithm analysis",
            "computational complexity",
            "formal",
            "mathematical",
        ],
        "general_ml": [],
    }

    keywords = plan.get("keywords") or []
    hypothesis = plan.get("hypothesis") or ""
    txt = " ".join([str(x) for x in keywords]) + " " + str(hypothesis) + " " + str(topic)
    t = txt.lower()

    scores: dict[str, int] = {}
    for domain, kws in domain_keywords.items():
        score = 0
        for kw in kws:
            if not kw:
                continue
            score += len(re.findall(re.escape(kw.lower()), t))
        scores[domain] = score

    max_score = max(scores.values())
    top = [d for d, s in scores.items() if s == max_score]
    if len(top) != 1:
        return "general_ml"
    return top[0]


def _load_template_experiments() -> dict[str, dict[str, Any]]:
    from aras.experiments.templates.nlp_experiments import EXPERIMENTS as NLP  # type: ignore
    from aras.experiments.templates.cv_experiments import EXPERIMENTS as CV  # type: ignore
    from aras.experiments.templates.rl_experiments import EXPERIMENTS as RL  # type: ignore
    from aras.experiments.templates.general_ml_experiments import EXPERIMENTS as GEN  # type: ignore
    from aras.experiments.templates.theory_experiments import EXPERIMENTS as THEORY  # type: ignore

    all_exps: list[dict[str, Any]] = []
    for lst in (NLP, CV, RL, GEN, THEORY):
        all_exps.extend(lst)

    out: dict[str, dict[str, Any]] = {}
    for exp in all_exps:
        name = str(exp.get("name") or "")
        if name:
            out[name] = exp
    return out


def _select_experiments_for_domain(domain: str) -> list[str]:
    domain_to_experiments: dict[str, list[str]] = {
        "nlp": ["exp1_text_classification", "exp2_sentiment_analysis", "exp3_tokenizer_benchmark"],
        "computer_vision": ["exp1_image_classification", "exp2_feature_extraction_benchmark", "exp3_data_augmentation_ablation"],
        "reinforcement_learning": ["exp1_bandit_comparison", "exp2_policy_gradient_cartpole", "exp3_q_learning_gridworld"],
        "theory": ["exp1_algorithm_complexity", "exp2_numerical_bounds_verification", "exp3_information_theory"],
        "general_ml": ["exp1_cross_validation_benchmark", "exp2_hyperparameter_sensitivity", "exp3_convergence_analysis"],
    }
    return domain_to_experiments.get(domain) or domain_to_experiments["general_ml"]


def _requires_internet(dataset_str: str) -> bool:
    s = (dataset_str or "").lower()
    return any(x in s for x in ["torchvision", "datasets", "20newsgroups", "fetch_olivetti_faces", "mnist", "imdb", "ag_news"])


def _tests_template() -> str:
    return '''from __future__ import annotations

import importlib.util
from pathlib import Path


def test_experiment_modules_import() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in root.glob("exp*.py"):
        spec = importlib.util.spec_from_file_location(p.stem, p)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
'''


def _summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    runs = results.get("runs") or []
    accs: list[float] = []
    wall: list[float] = []
    rss: list[float] = []
    for r in runs:
        m = r.get("metrics") or {}
        if "acc" in m:
            try:
                accs.append(float(m["acc"]))
            except Exception:
                pass
        wall.append(float(r.get("wall_seconds") or 0.0))
        rss.append(float(r.get("peak_rss_mb") or 0.0))
    return {
        "mean_acc": float(sum(accs) / max(1, len(accs))),
        "total_wall_seconds": float(sum(wall)),
        "mean_peak_rss_mb": float(sum(rss) / max(1, len(rss))),
    }


def _looks_like_unified_diff(s: str) -> bool:
    t = s.lstrip()
    return t.startswith("---") and "\n+++" in t and "\n@@" in t


def _apply_unified_diff_in_place(*, module_path: Path, diff_text: str) -> bool:
    if len(diff_text) > 20000:
        return False

    lines = diff_text.splitlines()
    try:
        i0 = next(i for i, l in enumerate(lines) if l.startswith("--- "))
        i1 = next(i for i, l in enumerate(lines) if l.startswith("+++ "))
    except StopIteration:
        return False
    if i1 < i0:
        return False

    if sum(1 for l in lines if l.startswith("--- ")) != 1:
        return False
    if sum(1 for l in lines if l.startswith("+++ ")) != 1:
        return False

    original = module_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    src_idx = 0

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("@@"):
            idx += 1
            continue
        m = re.match(r"@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)(?:,([0-9]+))?\s+@@", line)
        if not m:
            return False
        old_start = int(m.group(1))
        target_src = max(0, old_start - 1)
        if target_src < src_idx:
            return False
        out.extend(original[src_idx:target_src])
        src_idx = target_src
        idx += 1

        while idx < len(lines) and not lines[idx].startswith("@@") and not lines[idx].startswith("--- "):
            hl = lines[idx]
            if hl.startswith("\\ No newline at end of file"):
                idx += 1
                continue
            if not hl:
                return False
            tag = hl[0]
            text = hl[1:] if len(hl) > 1 else ""
            if tag == " ":
                if src_idx >= len(original) or original[src_idx] != text:
                    return False
                out.append(text)
                src_idx += 1
            elif tag == "-":
                if src_idx >= len(original) or original[src_idx] != text:
                    return False
                src_idx += 1
            elif tag == "+":
                out.append(text)
            else:
                return False
            idx += 1

    out.extend(original[src_idx:])
    module_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def _simplify_experiment_for_real_datasets(path: Path, *, failure_type: str) -> None:
    # Targeted shrinking based on patterns in our template scripts.
    txt = path.read_text(encoding="utf-8")
    if failure_type in {"timeout", "module_not_found", "import_error"}:
        txt = re.sub(r"n_train\s*=\s*2000", "n_train = 800", txt)
        txt = re.sub(r"n_test\s*=\s*500", "n_test = 200", txt)
        txt = re.sub(r"n_train\s*=\s*1000", "n_train = 500", txt)
        txt = re.sub(r"n_test\s*=\s*200", "n_test = 100", txt)
        txt = re.sub(r"n_docs\s*=\s*500", "n_docs = 200", txt)
        txt = re.sub(r"_load_dataset\(5000, 1000\)", "_load_dataset(1000, 300)", txt)
        txt = re.sub(r"_load_faces\(400\)", "_load_faces(200)", txt)
        txt = re.sub(r"steps\s*=\s*2000", "steps = 800", txt)
        txt = re.sub(r"trials\s*=\s*30", "trials = 10", txt)
        txt = re.sub(r"episodes\s*=\s*200", "episodes = 120", txt)
        txt = re.sub(r"episodes\s*=\s*1000", "episodes = 400", txt)
    path.write_text(txt, encoding="utf-8")


class CoderAgent(BaseAgent):
    """Design, implement, and run real experiments (with auto-debug attempts)."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="coder", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = FallbackRouter(settings=settings)
        self._failures: int = 0
        self._templates_by_name: dict[str, dict[str, Any]] | None = None

    def _ensure_templates_loaded(self) -> None:
        if self._templates_by_name is None:
            self._templates_by_name = _load_template_experiments()

    async def design_and_write_experiments(
        self,
        *,
        topic: str,
        plan: dict[str, Any],
        scraped: list[dict[str, Any]],
        output_root: Path,
    ) -> ExperimentBundle:
        slug = _slugify(topic)[:60]
        root = output_root / slug
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "__init__.py").write_text("", encoding="utf-8")

        self._ensure_templates_loaded()
        assert self._templates_by_name is not None

        domain = _classify_domain(plan=plan, topic=topic)
        selected_names = _select_experiments_for_domain(domain)

        modules: list[Path] = []
        for exp_name in selected_names:
            entry = self._templates_by_name.get(exp_name)
            if not entry:
                continue
            code = str(entry.get("code") or "")
            if not code.strip():
                continue
            code = _extract_code_placeholders(code, topic=topic, plan=plan, domain=domain)
            exp_file = root / f"{exp_name}.py"
            safe_write_text(exp_file, code)
            modules.append(exp_file)

        tests_dir = root / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        test_path = tests_dir / "test_experiments_import.py"
        safe_write_text(test_path, _tests_template())

        # Manifest.
        experiments_manifest: list[dict[str, Any]] = []
        total_est = 0.0
        requires_internet_any = False
        for exp_name in selected_names:
            entry = self._templates_by_name.get(exp_name, {})
            dataset = str(entry.get("dataset") or "")
            expected_metrics = entry.get("expected_metrics") or []
            estimated_runtime_s = float(entry.get("estimated_runtime_s") or (120.0 if domain == "theory" else 75.0))
            requires_internet_any = requires_internet_any or _requires_internet(dataset)
            total_est += estimated_runtime_s
            experiments_manifest.append(
                {
                    "name": exp_name,
                    "file": f"{exp_name}.py",
                    "domain": domain,
                    "dataset": dataset,
                    "expected_metrics": expected_metrics,
                    "estimated_runtime_s": estimated_runtime_s,
                }
            )

        manifest = {
            "topic": topic,
            "slug": slug,
            "domain": domain,
            "experiments": experiments_manifest,
            "total_estimated_runtime_s": float(total_est),
            "requires_internet": bool(requires_internet_any),
            "fallback_offline": True,
        }
        safe_write_text(root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        tests = [test_path]
        self.emit(f"Wrote experiments to {root} (domain={domain})")
        return ExperimentBundle(slug=slug, root=root, modules=modules, tests=tests)

    async def run_experiments(self, *, exp_bundle: ExperimentBundle, cycle: int) -> dict[str, Any]:
        results: dict[str, Any] = {"slug": exp_bundle.slug, "runs": []}
        for mod in exp_bundle.modules:
            run_dir = exp_bundle.root / mod.stem
            run_dir.mkdir(parents=True, exist_ok=True)

            target = run_dir / mod.name
            target.write_text(mod.read_text(encoding="utf-8"), encoding="utf-8")

            self.emit(f"Running {mod.name}")
            r = await self._run_with_autodebug(module_path=target, workdir=run_dir, cycle=cycle)
            results["runs"].append(r)

        results["summary"] = _summarize_results(results)
        safe_write_text(exp_bundle.root / "results.json", json.dumps(results, ensure_ascii=False, indent=2))
        self.emit("Experiments complete")
        return results

    async def _run_with_autodebug(self, *, module_path: Path, workdir: Path, cycle: int) -> dict[str, Any]:
        last: dict[str, Any] | None = None
        attempts = 0
        max_attempts = 4

        def _on_metric(obj: dict[str, Any]) -> None:
            try:
                event = {
                    "experiment": str(obj.get("experiment") or module_path.stem),
                    "key": str(obj.get("key") or "metric"),
                    "value": float(obj.get("value") or 0.0),
                    "step": int(obj.get("step") or 0),
                }
                self.emit("METRIC_EVENT " + json.dumps(event, ensure_ascii=False))
            except Exception:
                return

        while attempts < max_attempts:
            attempts += 1
            rr = await run_experiment_module(module_path=module_path, workdir=workdir, on_metric=_on_metric)
            d = rr.to_dict()
            last = d
            if rr.exit_code == 0:
                return d

            failure_text = (rr.stderr or rr.stdout or "")
            fail = classify_failure(error_text=failure_text)
            await self.memory.store_failure(
                failure={"agent_id": "coder", **fail, "context": {"experiment": module_path.stem, "exit_code": rr.exit_code}},
                cycle=cycle,
            )
            failure_type = str(fail.get("failure_type") or "unknown_error")
            self.emit(f"Experiment failure classified: {failure_type}", level="warning")
            self._failures += 1

            if attempts == 1:
                continue

            if attempts == 2:
                installed = await self._try_install_missing_imports(error_text=failure_text)
                if installed:
                    continue
                fixed = await self._attempt_fix(module_path=module_path, error_text=failure_text, escalation=False)
                if fixed:
                    continue

            if attempts == 3:
                _simplify_experiment_for_real_datasets(module_path, failure_type=failure_type)
                continue

            fixed = await self._attempt_fix(module_path=module_path, error_text=failure_text, escalation=True)
            if not fixed:
                _simplify_experiment_for_real_datasets(module_path, failure_type=failure_type)

        return last or {"name": module_path.stem, "exit_code": 1}

    async def _try_install_missing_imports(self, *, error_text: str) -> bool:
        low = (error_text or "").lower()
        m = re.search(r"no module named ['\"]([^'\"]+)['\"]", low)
        if not m:
            return False
        mod = m.group(1).split(".")[0].strip()
        allowed_import_to_pip: dict[str, str] = {
            "nltk": "nltk",
            "gymnasium": "gymnasium",
            "gym": "gym",
            "datasets": "datasets",
            "torchvision": "torchvision",
            "sklearn": "scikit-learn",
            "scipy": "scipy",
            "seaborn": "seaborn",
            "pandas": "pandas",
            "numpy": "numpy",
        }
        pip_pkg = allowed_import_to_pip.get(mod)
        if not pip_pkg:
            return False

        try:
            subprocess.run([sys.executable, "-m", "pip", "install", pip_pkg, "-q"], check=False)
            self.emit(f"Auto-installed missing dependency: {pip_pkg}", level="warning")
            return True
        except Exception:
            return False

    async def _attempt_fix(self, *, module_path: Path, error_text: str, escalation: bool = False) -> bool:
        if not error_text.strip():
            return False

        prefer = ["local", "nvidia", "openai"]
        model_overrides: dict[str, str] = {}
        if escalation or (self._failures > self.settings.routing_escalation_failures and self.settings.openai_api_key):
            prefer = ["openai", "nvidia", "local"]
            model_overrides["openai"] = self.settings.openai_escalation_model

        try:
            prompts = self.memory.current_prompts()
            rag = self.memory.rag_context(query="debug python experiment failures")
            system = f"{prompts.get('coder','')}\n\nRAG CONTEXT:\n{rag}\n\nReturn only a unified diff patch."
            user = (
                "Fix this Python experiment script to run successfully. Keep it minimal and reproducible.\n\n"
                f"ERROR:\n{error_text[:4000]}\n\n"
                f"FILE:\n{module_path.read_text(encoding='utf-8')[:6000]}"
            )
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="coder_autodebug",
                prefer=prefer,  # type: ignore[arg-type]
                model_overrides=model_overrides,  # type: ignore[arg-type]
                thinking=False,
                temperature=0.2,
                max_tokens=1200,
            )
            self.record_chat_result(res)
            try:
                self.add_tokens(res.tokens_used)
            except Exception:
                pass

            patch = (res.text or "").strip()
            if not _looks_like_unified_diff(patch):
                return False
            applied = _apply_unified_diff_in_place(module_path=module_path, diff_text=patch)
            if applied:
                self.emit("Applied LLM patch to experiment script")
            return applied
        except Exception:
            return False

