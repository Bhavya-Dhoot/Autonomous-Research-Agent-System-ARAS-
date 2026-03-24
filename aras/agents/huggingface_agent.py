from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from aras.agents.base import BaseAgent, EventSink
from aras.config import Settings
from aras.utils.logging import get_logger
from aras.utils.slugify import slugify


log = get_logger("hf-agent")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _extract_abstract(paper_tex: str) -> str:
    # Keep it LaTeX-ish but strip whitespace. Gradio Markdown renders LaTeX reasonably often.
    m = re.search(r"\\begin\\{abstract\\}([\\s\\S]*?)\\end\\{abstract\\}", paper_tex)
    if not m:
        return ""
    return " ".join(m.group(1).strip().split())


def _extract_sections(paper_tex: str) -> dict[str, str]:
    # Best-effort section extraction for Gradio display.
    # Uses IEEE template section names.
    names = ["Introduction", "Methodology", "Results", "Discussion", "Conclusion"]
    sections: dict[str, str] = {}
    for name in names:
        # Match \section{<Name>} ... up to next \section{...} or end of document.
        pat = rf"\\section\\{{{re.escape(name)}\\}}([\\s\\S]*?)(?=\\section\\{{|\\end\\{{document\\}})"
        m = re.search(pat, paper_tex)
        if not m:
            continue
        raw = m.group(1).strip()
        # Collapse whitespace to keep the payload light.
        sections[name.lower()] = " ".join(raw.split())
    # Normalize required keys.
    required_keys = ["introduction", "methodology", "results", "conclusion", "discussion"]
    out: dict[str, str] = {}
    for k in required_keys:
        out[k] = sections.get(k, "")
    return out


def _extract_paper_score(base_dir: Path) -> float:
    # Orchestrator appends "- Paper score: X" into IMPROVEMENT_LOG.md.
    imp = base_dir / "IMPROVEMENT_LOG.md"
    txt = _read_text(imp)
    matches = re.findall(r"- Paper score:\s*([0-9]+(?:\.[0-9]+)?)", txt)
    if not matches:
        return 0.0
    try:
        return float(matches[-1])
    except Exception:
        return 0.0


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_matching_files(src_dir: Path, dst_dir: Path, *, exts: set[str]) -> None:
    if not src_dir.exists():
        return
    _ensure_dir(dst_dir)
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(src_dir)
        target = dst_dir / rel
        _ensure_dir(target.parent)
        shutil.copy2(str(p), str(target))


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        _ensure_dir(dst.parent)
        shutil.copy2(str(src), str(dst))


def _find_results_json(outputs_root: Path, topic_slug: str) -> Path | None:
    # Expected: outputs_root/experiments/{slug}/results.json
    p = outputs_root / "experiments" / topic_slug / "results.json"
    if p.exists():
        return p
    # Fallback: scan for the newest results.json
    candidates = sorted(outputs_root.rglob("results.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    for c in candidates:
        # Avoid picking unrelated results.json files.
        if "experiments" in c.parts:
            return c
    return None


def _infer_experiments_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    # Supports both current and future results.json schemas.
    out: list[dict[str, Any]] = []
    if isinstance(results, dict) and isinstance(results.get("runs"), list):
        for r in results.get("runs") or []:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or r.get("metrics", {}).get("experiment") or "experiment")
            metrics = r.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            out.append({"name": name, "metrics": metrics})
    else:
        # Old/simplified schema: just summary at root.
        if isinstance(results, dict):
            out.append({"name": "summary", "metrics": results.get("summary") or results})
    return out


def _build_results_markdown_table(results: dict[str, Any], max_rows: int = 12) -> str:
    runs = _infer_experiments_from_results(results)
    # Extract numeric metrics.
    rows: list[list[str]] = []
    for run in runs[:max_rows]:
        name = str(run.get("name") or "")
        metrics = run.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        # Prefer accuracy-like metrics and otherwise take first numeric values.
        preferred = ["accuracy", "acc", "f1", "f1_macro", "final_loss", "roc_auc", "train_time_s", "inference_time_ms"]
        used = False
        for k in preferred:
            if k in metrics and isinstance(metrics[k], (int, float)):
                v = metrics[k]
                rows.append([name, k, f"{float(v):.4f}"])
                used = True
        if used:
            continue
        # fallback: top numeric metrics
        numeric = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
        numeric.sort(key=lambda kv: abs(float(kv[1])), reverse=True)
        for k, v in numeric[:3]:
            rows.append([name, str(k), f"{float(v):.4f}"])

    if not rows:
        rows = [["No metrics found", "", ""]]
    # Markdown table.
    lines = ["| Experiment | Metric | Value |", "|---|---:|---:|"]
    for exp, metric, val in rows:
        lines.append(f"| {exp} | {metric} | {val} |")
    return "\n".join(lines)


@dataclass(frozen=True)
class PublishedHFUrls:
    dataset_url: Optional[str]
    model_url: Optional[str]
    space_url: Optional[str]


class HuggingFaceAgent(BaseAgent):
    """Hugging Face publishing: dataset repo, optional model repo, and a Gradio Space."""

    def __init__(self, settings: Settings, on_event: EventSink, on_tokens=None, on_chat_result=None) -> None:
        super().__init__(agent_id="hf", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings

    async def publish_dataset(
        self,
        *,
        topic: str,
        outputs_root: Path,
        paper_pdf: Path | None,
        cycle: int,
    ) -> dict[str, Any]:
        """Publish artifacts to Hugging Face, best-effort.

        Returns a dict including dataset_url/model_url/space_url.
        For backward compatibility with the orchestrator's `hf_url` field, we also set `url = space_url`.
        """
        if not self.settings.hf_token or not self.settings.hf_username:
            self.emit("HF_TOKEN or HF_USERNAME missing; skipping HuggingFace publish", level="warning")
            return {"dataset_url": None, "model_url": None, "space_url": None, "url": None, "skipped": True}

        try:
            from huggingface_hub import CommitOperationAdd, HfApi, SpaceStage  # type: ignore
        except Exception as e:
            self.emit(f"huggingface_hub import failed; skipping HF publish: {e}", level="error")
            return {"dataset_url": None, "model_url": None, "space_url": None, "url": None, "skipped": True}

        api = HfApi(token=self.settings.hf_token)
        topic_slug = slugify(topic, max_length=60)
        base_dir = outputs_root
        paper_dir = base_dir / "paper"
        paper_tex_path = paper_dir / "paper.tex"
        paper_tex = _read_text(paper_tex_path)
        abstract = _extract_abstract(paper_tex)
        sections_dict = _extract_sections(paper_tex)
        paper_score = _extract_paper_score(base_dir)
        year = datetime.now(timezone.utc).year

        results_path = _find_results_json(outputs_root=base_dir, topic_slug=topic_slug)
        results: dict[str, Any] = {}
        if results_path and results_path.exists():
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except Exception:
                results = {}

        analysis_path = base_dir / "logs" / "analysis.json"
        analysis_text = _read_text(analysis_path)
        analysis_obj: dict[str, Any] = {}
        if analysis_text:
            try:
                analysis_obj = json.loads(analysis_text)
            except Exception:
                analysis_obj = {}

        github_url = ""  # best-effort: not currently persisted for HF publishing

        # Prepare output staging directory.
        tmp_root = base_dir / "_hf_tmp"
        try:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
        except Exception:
            pass
        _ensure_dir(tmp_root)

        published = PublishedHFUrls(dataset_url=None, model_url=None, space_url=None)

        # Step 1: dataset repo
        dataset_url: Optional[str] = None
        try:
            ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            dataset_repo_id = f"{self.settings.hf_username}/aras-{topic_slug}-{ts}"

            dataset_tmp = tmp_root / "dataset"
            try:
                if dataset_tmp.exists():
                    shutil.rmtree(dataset_tmp)
            except Exception:
                pass
            _ensure_dir(dataset_tmp)

            # Copy experiments artifacts (only what spec asked).
            experiments_dir = base_dir / "experiments"
            dst_experiments = dataset_tmp / "experiments"
            _copy_matching_files(
                experiments_dir,
                dst_experiments,
                exts={".py", ".json", ".png", ".pdf"},
            )

            # Paper artifacts.
            dst_paper = dataset_tmp / "paper"
            _copy_matching_files(paper_dir, dst_paper, exts={".pdf", ".tex"})
            if paper_pdf and paper_pdf.exists():
                _copy_file_if_exists(paper_pdf, dst_paper / paper_pdf.name)

            # Logs.
            dst_logs = dataset_tmp / "logs"
            _copy_file_if_exists(analysis_path, dst_logs / "analysis.json")
            _copy_file_if_exists(base_dir / "logs" / "cost_report.json", dst_logs / "cost_report.json")

            # Memory snapshots.
            _copy_file_if_exists(base_dir / "memory_snapshot" / "chroma_db.zip", dataset_tmp / "memory_snapshot" / "chroma_db.zip")

            # Dataset README.
            desc = abstract[:300] + ("..." if len(abstract) > 300 else "")
            experiments_list = _infer_experiments_from_results(results)
            exp_lines: list[str] = []
            for exp in experiments_list[:6]:
                name = exp.get("name") or "experiment"
                metrics = exp.get("metrics") or {}
                if not isinstance(metrics, dict):
                    metrics = {}
                numeric_metrics = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
                numeric_metrics.sort(key=lambda kv: abs(float(kv[1])), reverse=True)
                take = numeric_metrics[:4]
                if take:
                    metric_str = ", ".join([f"{k}={float(v):.4f}" for k, v in take])
                else:
                    metric_str = "metrics unavailable"
                exp_lines.append(f"- {name}: {metric_str}")
            exp_block = "\n".join(exp_lines) if exp_lines else "- No experiments found"

            paper_link = "paper.pdf not uploaded"
            if (dst_paper / "paper.pdf").exists():
                paper_link = "[paper.pdf](paper/paper.pdf)"

            bib_url = f"https://huggingface.co/datasets/{self.settings.hf_username}/aras-{topic_slug}"
            dataset_readme = f"""---
license: mit
task_categories: [other]
tags: [autonomous-research, aras, {topic_slug}]
---
# ARAS Dataset: {topic}
## Description
{desc}
## Experiments
{exp_block}
## Paper
{paper_link}
## Citation
```bibtex
@misc{{aras_{topic_slug}_{year},
  title={{{topic}}},
  author={{ARAS Autonomous Research System}},
  year={{{year}}},
  url={{{bib_url}}}
}}
```
"""
            (dataset_tmp / "README.md").write_text(dataset_readme, encoding="utf-8")

            self.emit(f"Creating HF dataset repo: {dataset_repo_id}")
            api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=False, exist_ok=True)
            self.emit("Uploading HF dataset files...")
            api.upload_folder(
                folder_path=str(dataset_tmp),
                repo_id=dataset_repo_id,
                repo_type="dataset",
                path_in_repo="",
                commit_message=f"ARAS dataset artifacts ({topic_slug})",
            )
            dataset_url = f"https://huggingface.co/datasets/{dataset_repo_id}"
            self.emit(f"HF dataset published: {dataset_url}")
        except Exception as e:
            self.emit(f"HF dataset publish failed: {e}", level="error")

        # Step 2: optional model repo
        model_url: Optional[str] = None
        try:
            experiments_dir = base_dir / "experiments"
            checkpoint_exts = {".pt", ".pkl", ".joblib", ".safetensors"}
            ckpts: list[Path] = []
            if experiments_dir.exists():
                for p in experiments_dir.rglob("*"):
                    if p.is_file() and p.suffix.lower() in checkpoint_exts:
                        ckpts.append(p)

            if ckpts:
                model_repo_id = f"{self.settings.hf_username}/aras-{topic_slug}-model"
                model_tmp = tmp_root / "model"
                try:
                    if model_tmp.exists():
                        shutil.rmtree(model_tmp)
                except Exception:
                    pass
                _ensure_dir(model_tmp)

                dst_ckpt_dir = model_tmp / "checkpoints"
                _ensure_dir(dst_ckpt_dir)
                for ck in ckpts:
                    dst = dst_ckpt_dir / ck.name
                    shutil.copy2(str(ck), str(dst))

                # Try to extract methodology snippet.
                methodology_snip = ""
                m = re.search(r"\\section\\{Methodology\\}([\\s\\S]*?)(?=\\section\\{|\\end\\{document\\})", paper_tex)
                if m:
                    methodology_snip = " ".join(m.group(1).strip().split())[:500]
                if not methodology_snip:
                    methodology_snip = "Generated autonomously by ARAS v2."

                library_name = "pytorch" if any(p.suffix.lower() == ".pt" for p in ckpts) else "sklearn"
                # Metrics table from results.json.
                metrics_md = _build_results_markdown_table(results)

                usage_snippet = (
                    "```python\n"
                    "# Load and use your model checkpoint\n"
                    f"# (Checkpoint files uploaded by ARAS)\n"
                    "from pathlib import Path\n"
                    "ckpt = Path('checkpoints')\n"
                    "print('Available checkpoints:', [p.name for p in ckpt.iterdir()])\n"
                    "```\n"
                )

                model_card = f"""---
license: mit
library_name: {library_name}
tags: [autonomous-research, {topic_slug}]
---
# ARAS Model: {topic}
## Description
{methodology_snip}
## Usage
{usage_snippet}
## Metrics
{metrics_md}
"""
                (model_tmp / "README.md").write_text(model_card, encoding="utf-8")

                self.emit(f"Creating HF model repo: {model_repo_id}")
                api.create_repo(repo_id=model_repo_id, repo_type="model", private=False, exist_ok=True)
                self.emit("Uploading HF model files...")
                api.upload_folder(
                    folder_path=str(model_tmp),
                    repo_id=model_repo_id,
                    repo_type="model",
                    path_in_repo="",
                    commit_message=f"ARAS model checkpoint artifacts ({topic_slug})",
                )
                model_url = f"https://huggingface.co/{model_repo_id}"
                self.emit(f"HF model published: {model_url}")
            else:
                self.emit("No model checkpoints detected; skipping HF model publish", level="warning")
        except Exception as e:
            self.emit(f"HF model publish failed: {e}", level="error")

        # Step 3: Gradio Space
        space_url: Optional[str] = None
        try:
            space_repo_id = f"{self.settings.hf_username}/aras-{topic_slug}-demo"
            # Build Space repo staging.
            space_tmp = tmp_root / "space"
            try:
                if space_tmp.exists():
                    shutil.rmtree(space_tmp)
            except Exception:
                pass
            _ensure_dir(space_tmp)

            # Figures: gather PNGs from paper/figures.
            figs_src = paper_dir / "figures"
            pngs: list[Path] = []
            if figs_src.exists():
                for p in figs_src.rglob("*.png"):
                    if p.is_file():
                        pngs.append(p)
            pngs = sorted(pngs, key=lambda p: p.stat().st_mtime, reverse=True)[:20]

            fig_flat_paths: list[str] = []
            for i, p in enumerate(pngs, start=1):
                out_name = f"fig_{i:03d}.png"
                dst = space_tmp / out_name
                shutil.copy2(str(p), str(dst))
                fig_flat_paths.append(out_name)

            # Bundle results.json (required) + analysis.json.
            if results_path and results_path.exists():
                shutil.copy2(str(results_path), str(space_tmp / "results.json"))
            if analysis_path.exists():
                shutil.copy2(str(analysis_path), str(space_tmp / "analysis.json"))

            # Compose a minimal sections dict injection.
            sections_dict_json = json.dumps(sections_dict, ensure_ascii=False)
            abstract_escaped = abstract.replace('"""', '\\"\\"\\"')
            dataset_url_for_app = ""
            if dataset_url:
                dataset_url_for_app = dataset_url

            app_py = """import gradio as gr
import json
from pathlib import Path

# ── Load research artifacts ────────────────────────────────────────────
TOPIC = __TOPIC__
ABSTRACT = __ABSTRACT__
PAPER_SCORE = __PAPER_SCORE__
GITHUB_URL = __GITHUB_URL__
DATASET_URL = __DATASET_URL__

def load_results():
    path = Path("results.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {{}}

def load_analysis():
    path = Path("analysis.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {{}}

results = load_results()
analysis = load_analysis()

# ── Build metrics table ────────────────────────────────────────────────
def get_metrics_table():
    rows = []
    runs = results.get("runs", [])
    for run in runs:
        name = run.get("name", "experiment")
        metrics = run.get("metrics", {{}}) or {{}}
        if not isinstance(metrics, dict):
            metrics = {{}}
        for k, v in metrics.items():
            try:
                if isinstance(v, float) or isinstance(v, int):
                    rows.append([name, k, f"{{float(v):.4f}}"])
                else:
                    rows.append([name, k, str(v)])
            except Exception:
                rows.append([name, k, str(v)])
    if not rows:
        rows = [["No experiments found", "", ""]]
    return rows

# ── Figure loader ────────────────────────────────────────────────────────
def get_figures():
    figs = []
    for ext in ["*.png"]:
        figs.extend(Path(".").rglob(ext))
    return [str(f) for f in sorted(figs)[:12]]  # max 12 figures

# ── Paper sections ───────────────────────────────────────────────────────
SECTIONS = __SECTIONS_JSON__  # populated by ARAS at generation time

def get_section(section_name):
    return SECTIONS.get(section_name, "Section not available.")

# ── Gradio UI ──────────────────────────────────────────────────────────
with gr.Blocks(
    title=f"ARAS Research: {TOPIC}",
    theme=gr.themes.Soft(),
    css=\"\"\"
.metric-table { font-family: monospace; }
.score-badge { font-size: 2em; font-weight: bold; color: #2563eb; }
\"\"\",
) as demo:
    gr.Markdown(f"# 🔬 ARAS Research: {TOPIC}")
    gr.Markdown(f"**Paper Score:** <span class='score-badge'>{PAPER_SCORE}/10</span>")

    with gr.Tabs():
        with gr.Tab("📄 Abstract"):
            gr.Markdown(f"## Abstract\\n\\n{ABSTRACT}")
            with gr.Row():
                if GITHUB_URL:
                    gr.Markdown(f"**GitHub:** [repo]({GITHUB_URL})")
                else:
                    gr.Markdown("**GitHub:** Not published")
                if DATASET_URL:
                    gr.Markdown(f"**Dataset:** [repo]({DATASET_URL})")
                else:
                    gr.Markdown("**Dataset:** Not published")

        with gr.Tab("📊 Results"):
            gr.Markdown("## Experiment Results")
            gr.DataFrame(
                value=get_metrics_table(),
                headers=["Experiment", "Metric", "Value"],
                label="Metrics",
                elem_classes=["metric-table"],
            )
            gr.Markdown("### Analysis\\n\\n" + str(analysis.get("narrative", "No analysis available.")))

        with gr.Tab("🖼️ Figures"):
            figs = get_figures()
            if figs:
                gr.Gallery(
                    value=figs,
                    label="Experiment Figures",
                    columns=3,
                    height="auto",
                )
            else:
                gr.Markdown("No figures available.")

        with gr.Tab("📖 Paper Sections"):
            section_choice = gr.Dropdown(
                choices=list(SECTIONS.keys()),
                label="Select Section",
                value=list(SECTIONS.keys())[0] if SECTIONS else None,
            )
            section_content = gr.Textbox(
                label="Section Content",
                lines=20,
                interactive=False,
            )
            section_choice.change(
                fn=get_section,
                inputs=section_choice,
                outputs=section_content,
            )

        with gr.Tab("ℹ️ About ARAS"):
            gr.Markdown(
                \"\"\"## About ARAS
This paper and its experiments were generated autonomously by
**ARAS (Autonomous Research Agent System) v2**.
\"\"\"
            )

if __name__ == "__main__":
    demo.launch()
"""

            app_py = app_py.replace("__TOPIC__", json.dumps(topic))
            app_py = app_py.replace("__ABSTRACT__", json.dumps(abstract_escaped))
            app_py = app_py.replace("__PAPER_SCORE__", str(float(paper_score)))
            app_py = app_py.replace("__GITHUB_URL__", json.dumps(github_url))
            app_py = app_py.replace("__DATASET_URL__", json.dumps(dataset_url_for_app))
            app_py = app_py.replace("__SECTIONS_JSON__", sections_dict_json)

            (space_tmp / "app.py").write_text(app_py, encoding="utf-8")
            (space_tmp / "requirements.txt").write_text("gradio>=4.0.0\n", encoding="utf-8")

            # requirements for spaces also often need no extra deps; rely on gradio defaults.

            self.emit(f"Creating HF Space: {space_repo_id}")
            api.create_repo(repo_id=space_repo_id, repo_type="space", private=False, exist_ok=True, space_sdk="gradio", space_hardware="cpu-basic")

            self.emit("Uploading HF Space files...")
            api.upload_folder(
                folder_path=str(space_tmp),
                repo_id=space_repo_id,
                repo_type="space",
                path_in_repo="",
                commit_message=f"ARAS demo app ({topic_slug})",
            )

            # Wait for build/run stage up to 60s.
            for _ in range(12):
                try:
                    info = api.space_info(space_repo_id)
                    stage = info.stage
                    if stage == SpaceStage.RUNNING or str(stage) == "RUNNING":
                        break
                except Exception:
                    pass
                time.sleep(5)

            space_url = f"https://huggingface.co/spaces/{space_repo_id}"
            self.emit(f"HF Space ready: {space_url}")
        except Exception as e:
            self.emit(f"HF Space publish failed: {e}", level="error")

        # Persist last-publish artifact for UI consumption.
        try:
            urls_payload = {
                "dataset_url": dataset_url,
                "model_url": model_url,
                "space_url": space_url,
            }
            _ensure_dir(base_dir / "logs")
            (base_dir / "logs" / "hf_last.json").write_text(_json_dumps(urls_payload), encoding="utf-8")
            self.emit("HF_EVENT " + _json_dumps(urls_payload))
        except Exception:
            pass

        # Cleanup tmp files.
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

        ret = {
            "dataset_url": dataset_url,
            "model_url": model_url,
            "space_url": space_url,
            # Orchestrator expects `url` to populate `hf_url` in UI state.
            "url": space_url,
            "skipped": False,
        }
        return ret

