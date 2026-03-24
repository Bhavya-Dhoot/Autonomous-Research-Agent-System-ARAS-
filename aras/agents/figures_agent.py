from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from aras.agents.base import BaseAgent, EventSink
from aras.config import Settings
from aras.utils.logging import get_logger


log = get_logger("figures-agent")


class FiguresAgent(BaseAgent):
    """Generate publication-quality figures and reusable LaTeX snippets."""

    def __init__(self, settings: Settings, on_event: EventSink, on_tokens=None, on_chat_result=None) -> None:
        super().__init__(agent_id="figures", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings

    async def generate(self, *, topic: str, results: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Generate figures and return analysis augmentation."""
        figures_dir = output_dir / "figures"
        self._reset_figures_dir(figures_dir)

        runs = list(results.get("runs") or [])
        curve_series: list[tuple[str, list[float]]] = []
        perf_points: list[tuple[str, float]] = []
        copied_experiment_figs: list[tuple[str, str]] = []
        health = {"success": 0, "degraded": 0, "failed": 0}

        sns.set_theme(style="whitegrid")
        fig_paths: dict[str, str] = {}
        figures_latex_parts: list[str] = []
        figure_inventory: list[dict[str, Any]] = []
        paper_eligible_figures: list[str] = []

        captions: dict[str, dict[str, Any]] = {}

        for idx, run in enumerate(runs, start=1):
            name = self._run_name(run=run, fallback=f"run_{idx}")
            metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
            is_failed = self._is_failed_run(run=run, metrics=metrics)
            if is_failed:
                health["failed"] += 1
            elif self._is_degraded_run(metrics=metrics):
                health["degraded"] += 1
            else:
                health["success"] += 1

            losses = self._extract_losses(metrics)
            if len(losses) >= 2 and not is_failed:
                curve_series.append((name, losses))

            score = self._extract_performance_score(metrics)
            if score is not None and not is_failed:
                perf_points.append((name, score))

            copied = self._copy_run_artifact_figure(run=run, figures_dir=figures_dir, allow_copy=(not is_failed))
            if copied is not None:
                rel_path, caption = copied
                copied_experiment_figs.append(copied)
                fig_id = Path(rel_path).stem
                captions[fig_id] = {
                    "caption": caption,
                    "source_type": "artifact",
                    "confidence": "high",
                    "reason": "native_experiment_artifact",
                }

        if perf_points:
            names = [n for n, _ in perf_points]
            scores = [v for _, v in perf_points]
            plt.figure(figsize=(7.2, 4.2))
            bars = plt.bar(names, scores, color="#3A6EA5")
            plt.xlabel("Experiment", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Performance overview for: {topic}", fontsize=13)
            plt.ylim(0.0, max(1.0, max(scores) * 1.1))
            for b, score in zip(bars, scores):
                plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{score:.3f}", ha="center", va="bottom", fontsize=8)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            perf_paths = self._save_dual_format(figures_dir=figures_dir, stem="performance_overview")
            plt.close()
            fig_paths["performance_overview_png"] = str(perf_paths["png"].resolve())
            fig_paths["performance_overview_pdf"] = str(perf_paths["pdf"].resolve())
            self._register_figure(
                inventory=figure_inventory,
                paper_eligible=paper_eligible_figures,
                latex_parts=figures_latex_parts,
                fig_id="performance_overview",
                file_name="performance_overview.pdf",
                caption=f"Comparable performance score by run for {self._escape_latex(topic)}.",
                label="fig:performance_overview",
                source_type="aggregate",
                confidence="high",
                reason="comparable_score_from_successful_runs",
                include_in_paper=True,
            )
            captions["performance_overview"] = {
                "caption": "Comparable experiment score by run.",
                "source_type": "aggregate",
                "confidence": "high",
                "reason": "comparable_score_from_successful_runs",
            }

        if curve_series:
            plt.figure(figsize=(7.2, 4.2))
            for name, losses in curve_series:
                xs = list(range(1, len(losses) + 1))
                plt.plot(xs, losses, linewidth=2.0, label=name)
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Training loss", fontsize=12)
            plt.title(f"Training curves for: {topic}", fontsize=13)
            if len(curve_series) <= 8:
                plt.legend(fontsize=8)
            plt.tight_layout()
            curve_paths = self._save_dual_format(figures_dir=figures_dir, stem="loss_curves")
            plt.close()
            fig_paths["loss_curves_png"] = str(curve_paths["png"].resolve())
            fig_paths["loss_curves_pdf"] = str(curve_paths["pdf"].resolve())
            self._register_figure(
                inventory=figure_inventory,
                paper_eligible=paper_eligible_figures,
                latex_parts=figures_latex_parts,
                fig_id="loss_curves",
                file_name="loss_curves.pdf",
                caption=f"Training loss curves across experiments for {self._escape_latex(topic)}.",
                label="fig:loss_curves",
                source_type="aggregate",
                confidence="high",
                reason="valid_multistep_loss_series",
                include_in_paper=True,
            )
            captions["loss_curves"] = {
                "caption": "Multi-step training loss curves from valid experiment runs.",
                "source_type": "aggregate",
                "confidence": "high",
                "reason": "valid_multistep_loss_series",
            }
        else:
            plt.figure(figsize=(7.2, 3.6))
            plt.axis("off")
            plt.text(
                0.5,
                0.6,
                "No valid multi-step loss series were produced.",
                ha="center",
                va="center",
                fontsize=12,
            )
            plt.text(
                0.5,
                0.42,
                "This cycle included scalar-only, failed, or placeholder outputs.",
                ha="center",
                va="center",
                fontsize=10,
            )
            plt.tight_layout()
            diag_paths = self._save_dual_format(figures_dir=figures_dir, stem="training_data_quality")
            plt.close()
            fig_paths["training_data_quality_png"] = str(diag_paths["png"].resolve())
            fig_paths["training_data_quality_pdf"] = str(diag_paths["pdf"].resolve())
            self._register_figure(
                inventory=figure_inventory,
                paper_eligible=paper_eligible_figures,
                latex_parts=figures_latex_parts,
                fig_id="training_data_quality",
                file_name="training_data_quality.pdf",
                caption="Diagnostic note: no valid multi-step loss series were available this cycle.",
                label="fig:training_data_quality",
                source_type="diagnostic",
                confidence="low",
                reason="no_valid_multistep_series",
                include_in_paper=False,
            )
            captions["training_data_quality"] = {
                "caption": "Diagnostic note for missing valid time-series curves.",
                "source_type": "diagnostic",
                "confidence": "low",
                "reason": "no_valid_multistep_series",
            }

        plt.figure(figsize=(6.8, 3.8))
        health_labels = ["success", "degraded", "failed"]
        health_values = [int(health[k]) for k in health_labels]
        colors = ["#2E8B57", "#C89B3C", "#B03A2E"]
        bars = plt.bar(health_labels, health_values, color=colors)
        for b, v in zip(bars, health_values):
            plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), str(v), ha="center", va="bottom", fontsize=9)
        plt.title("Run health summary", fontsize=12)
        plt.ylabel("Runs", fontsize=11)
        plt.tight_layout()
        health_paths = self._save_dual_format(figures_dir=figures_dir, stem="run_health")
        plt.close()
        fig_paths["run_health_png"] = str(health_paths["png"].resolve())
        fig_paths["run_health_pdf"] = str(health_paths["pdf"].resolve())
        self._register_figure(
            inventory=figure_inventory,
            paper_eligible=paper_eligible_figures,
            latex_parts=figures_latex_parts,
            fig_id="run_health",
            file_name="run_health.pdf",
            caption="Run health distribution across the cycle (success, degraded, failed).",
            label="fig:run_health",
            source_type="aggregate",
            confidence="high",
            reason="health_distribution_from_runs",
            include_in_paper=True,
        )
        captions["run_health"] = {
            "caption": "Counts of successful, degraded, and failed runs.",
            "source_type": "aggregate",
            "confidence": "high",
            "reason": "health_distribution_from_runs",
        }

        if copied_experiment_figs:
            for rel_path, caption in copied_experiment_figs[:3]:
                label = "fig:exp_" + self._slug_stem(Path(rel_path).stem)
                rel_norm = rel_path.replace("\\", "/")
                fig_id = Path(rel_norm).stem
                self._register_figure(
                    inventory=figure_inventory,
                    paper_eligible=paper_eligible_figures,
                    latex_parts=figures_latex_parts,
                    fig_id=fig_id,
                    file_name=rel_norm,
                    caption=self._escape_latex(caption),
                    label=label,
                    source_type="artifact",
                    confidence="high",
                    reason="native_experiment_artifact",
                    include_in_paper=True,
                )

        if captions:
            (figures_dir / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")

        low_conf_count = sum(1 for item in figure_inventory if str(item.get("confidence")) == "low")
        high_conf_count = sum(1 for item in figure_inventory if str(item.get("confidence")) == "high")
        all_runs_degraded = bool(runs) and int(health["success"]) == 0

        architecture_tikz = self._architecture_tikz_snippet()

        return {
            "architecture_tikz": architecture_tikz,
            "figures_latex": "\n\n".join(figures_latex_parts),
            "figures_paths": fig_paths,
            "figure_inventory": figure_inventory,
            "paper_eligible_figures": paper_eligible_figures,
            "figure_quality_summary": {
                "high_confidence": high_conf_count,
                "low_confidence": low_conf_count,
                "excluded_from_paper": low_conf_count,
                "all_runs_degraded": all_runs_degraded,
                "health": health,
            },
        }

    def _reset_figures_dir(self, figures_dir: Path) -> None:
        if figures_dir.exists():
            for p in list(figures_dir.iterdir()):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        figures_dir.mkdir(parents=True, exist_ok=True)

    def _run_name(self, *, run: dict[str, Any], fallback: str) -> str:
        name = str(run.get("name") or "").strip()
        if name:
            return name
        raw_metrics = run.get("metrics")
        metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
        exp = str(metrics.get("experiment") or "").strip()
        return exp or fallback

    def _extract_losses(self, metrics: dict[str, Any]) -> list[float]:
        raw = metrics.get("losses")
        if not isinstance(raw, list):
            return []
        vals: list[float] = []
        for x in raw:
            if self._is_finite_number(x):
                vals.append(float(x))
        return vals

    def _extract_performance_score(self, metrics: dict[str, Any]) -> float | None:
        candidates: list[float | None] = []
        nested = metrics.get("metrics")
        if isinstance(nested, dict):
            for k in ("best_accuracy", "best_cv_accuracy", "mean_accuracy", "accuracy", "f1_macro"):
                candidates.append(self._normalize_accuracy_like_score(nested.get(k)))

        for k in ("acc", "accuracy", "f1_macro"):
            candidates.append(self._normalize_accuracy_like_score(metrics.get(k)))

        for c in candidates:
            if c is not None:
                return float(c)
        return None

    def _normalize_accuracy_like_score(self, x: Any) -> float | None:
        if not self._is_finite_number(x):
            return None
        v = float(x)
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            scaled = v / 100.0
            if 0.0 <= scaled <= 1.0:
                return scaled
        return None

    def _is_failed_run(self, *, run: dict[str, Any], metrics: dict[str, Any]) -> bool:
        if int(run.get("exit_code") or 0) != 0:
            return True
        if metrics.get("error"):
            return True
        if bool(metrics.get("timed_out")):
            return True
        return False

    def _is_degraded_run(self, *, metrics: dict[str, Any]) -> bool:
        losses = self._extract_losses(metrics)
        if len(losses) == 1 and losses[0] in (0.0, 1.0):
            return True
        return False

    def _copy_run_artifact_figure(self, *, run: dict[str, Any], figures_dir: Path, allow_copy: bool) -> tuple[str, str] | None:
        if not allow_copy:
            return None
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        raw = artifacts.get("loss.png")
        if not isinstance(raw, str) or not raw.strip():
            return None
        src = Path(raw)
        if not src.exists() or not src.is_file():
            return None
        exp_name = self._slug_stem(str(run.get("name") or src.parent.name or "experiment"))
        dst_rel = Path("experiments") / f"{exp_name}.png"
        dst = figures_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(src), str(dst))
        except Exception:
            return None
        caption = f"Native experiment figure for {run.get('name') or exp_name}."
        return str(dst_rel).replace("\\", "/"), caption

    def _register_figure(
        self,
        *,
        inventory: list[dict[str, Any]],
        paper_eligible: list[str],
        latex_parts: list[str],
        fig_id: str,
        file_name: str,
        caption: str,
        label: str,
        source_type: str,
        confidence: str,
        reason: str,
        include_in_paper: bool,
    ) -> None:
        item = {
            "id": fig_id,
            "file": file_name,
            "caption": caption,
            "label": label,
            "source_type": source_type,
            "confidence": confidence,
            "reason": reason,
            "paper_included": bool(include_in_paper and confidence == "high"),
        }
        inventory.append(item)
        if item["paper_included"]:
            paper_eligible.append(fig_id)
            latex_parts.append(self._latex_figure_snippet(file_name=file_name, caption=caption, label=label))

    def _save_dual_format(self, *, figures_dir: Path, stem: str) -> dict[str, Path]:
        out_base = figures_dir / stem
        png = out_base.with_suffix(".png")
        pdf = out_base.with_suffix(".pdf")
        plt.savefig(str(png), dpi=220)
        plt.savefig(str(pdf))
        return {"png": png, "pdf": pdf}

    def _latex_figure_snippet(self, *, file_name: str, caption: str, label: str) -> str:
        return (
            "\\begin{figure}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{file_name}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}"
        )

    def _is_finite_number(self, x: Any) -> bool:
        try:
            v = float(x)
        except Exception:
            return False
        return v == v and v not in (float("inf"), float("-inf"))

    def _slug_stem(self, s: str) -> str:
        out = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
        return out or "figure"

    def _architecture_tikz_snippet(self) -> str:
        # Keep it compact and reusable inside the LaTeX architecture subsection.
        return (
            "\\begin{tikzpicture}[node distance=12mm, >=Latex]\n"
            "\\tikzstyle{box}=[draw,rounded corners,align=center,minimum height=7mm,minimum width=25mm]\n"
            "\\node[box] (o) {Orchestrator};\n"
            "\\node[box, right=of o] (n) {Novelty};\n"
            "\\node[box, below=of n] (s) {Scraping};\n"
            "\\node[box, below=of s] (c) {Citations};\n"
            "\\node[box, left=of c] (e) {Experiments};\n"
            "\\node[box, left=of e] (w) {Writing};\n"
            "\\node[box, above=of s] (m) {Memory};\n"
            "\\draw[->] (o) -- (n);\n"
            "\\draw[->] (o) -- (s);\n"
            "\\draw[->] (s) -- (c);\n"
            "\\draw[->] (c) -- (e);\n"
            "\\draw[->] (e) -- (m);\n"
            "\\draw[->] (m) -- (w);\n"
            "\\end{tikzpicture}"
        )

    def _escape_latex(self, s: str) -> str:
        # Minimal escaping for captions.
        return (
            s.replace("\\", "\\textbackslash{}")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("_", "\\_")
        )
