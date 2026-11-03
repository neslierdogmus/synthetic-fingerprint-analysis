from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VOLUME_LABELS: dict[str, str] = {
    "V0": "0N", "V1": "1N", "V2": "3N", "V3": "7N",
}
_SUBST_LABELS: dict[str, str] = {
    "R0": "100%", "R1": "75%", "R2": "50%", "R3": "25%", "R4": "0%",
}
_METRIC_LABELS: dict[str, str] = {
    "fc":  "Top-1 Accuracy",
    "fmd": "F1 Score",
    "foe": "Mean Angular Error (°)",
}
_FC_CLASSES = ["T", "A", "L", "R", "W"]


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 14,
    })


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _top_sources(sub: pd.DataFrame, condition_col: str, condition_val: str,
                 n: int = 3) -> list[str]:
    """Return up to n sources ranked by mean_metric at a given condition."""
    at_cond = sub[sub[condition_col] == condition_val]
    if at_cond.empty:
        at_cond = sub
    return (
        at_cond.groupby("source")["mean_metric"]
        .mean()
        .nlargest(n)
        .index.tolist()
    )


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def _plot_volume_scaling(df: pd.DataFrame, out: Path) -> None:
    """Plot 1: Volume scaling curves per (task, model)."""
    vol_order = ["0N", "1N", "3N", "7N"]
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model"].unique()):
            sub = df[
                (df["task"] == task) & (df["model"] == model)
                & df["condition"].str.startswith("V")
            ].copy()
            if sub.empty:
                continue

            sub["vol_label"] = sub["condition"].map(_VOLUME_LABELS)
            top_srcs = _top_sources(sub, "condition", "V3")

            fig, ax = plt.subplots(figsize=(10, 6))
            for src in top_srcs:
                s = (
                    sub[sub["source"] == src]
                    .groupby("vol_label")[["mean_metric", "std_metric"]]
                    .mean()
                    .reindex(vol_order)
                    .reset_index()
                )
                ax.errorbar(
                    s["vol_label"], s["mean_metric"],
                    yerr=s["std_metric"], label=src,
                    marker="o", capsize=4,
                )
            ax.set_xlabel("Synthetic Volume")
            ax.set_ylabel(_METRIC_LABELS.get(task, "Metric"))
            ax.set_title(f"Volume Scaling — {task.upper()} {model}")
            ax.legend(title="Source", bbox_to_anchor=(1.05, 1), loc="upper left")
            _save_figure(fig, out, f"volume_scaling_{task}_{model}")


def _plot_substitution(df: pd.DataFrame, out: Path) -> None:
    """Plot 2: Substitution curves per (task, model)."""
    subst_order = ["100%", "75%", "50%", "25%", "0%"]
    for task in sorted(df["task"].unique()):
        for model in sorted(df["model"].unique()):
            sub = df[
                (df["task"] == task) & (df["model"] == model)
                & df["condition"].str.startswith("R")
            ].copy()
            if sub.empty:
                continue

            sub["subst_label"] = sub["condition"].map(_SUBST_LABELS)
            top_srcs = _top_sources(sub, "condition", "R4")

            fig, ax = plt.subplots(figsize=(10, 6))
            for src in top_srcs:
                s = (
                    sub[sub["source"] == src]
                    .groupby("subst_label")[["mean_metric", "std_metric"]]
                    .mean()
                    .reindex(subst_order)
                    .reset_index()
                )
                ax.errorbar(
                    s["subst_label"], s["mean_metric"],
                    yerr=s["std_metric"], label=src,
                    marker="o", capsize=4,
                )
            ax.set_xlabel("Real Data Fraction")
            ax.set_ylabel(_METRIC_LABELS.get(task, "Metric"))
            ax.set_title(f"Substitution — {task.upper()} {model}")
            ax.legend(title="Source", bbox_to_anchor=(1.05, 1), loc="upper left")
            _save_figure(fig, out, f"substitution_{task}_{model}")


def _plot_source_comparison(df: pd.DataFrame, out: Path) -> None:
    """Plot 3: Source comparison bar chart (grouped by model) at V1."""
    models = sorted(df["model"].unique())
    width = 0.8 / max(len(models), 1)

    for task in sorted(df["task"].unique()):
        sub_v1 = df[(df["task"] == task) & (df["condition"] == "V1")]
        if sub_v1.empty:
            continue

        sources = sorted(sub_v1["source"].unique())
        x = np.arange(len(sources))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, model in enumerate(models):
            sub_m = sub_v1[sub_v1["model"] == model]
            means = [
                float(sub_m[sub_m["source"] == s]["mean_metric"].mean())
                if not sub_m[sub_m["source"] == s].empty else float("nan")
                for s in sources
            ]
            stds = [
                float(sub_m[sub_m["source"] == s]["std_metric"].mean())
                if not sub_m[sub_m["source"] == s].empty else 0.0
                for s in sources
            ]
            ax.bar(
                x + i * width, means, width,
                label=model, yerr=stds, capsize=3,
            )

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(sources, rotation=45, ha="right")
        ax.set_xlabel("Synthetic Source")
        ax.set_ylabel(_METRIC_LABELS.get(task, "Metric"))
        ax.set_title(f"Source Comparison at V1 — {task.upper()}")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        _save_figure(fig, out, f"source_comparison_{task}")


def _plot_model_comparison(df: pd.DataFrame, out: Path) -> None:
    """Plot 4: Model comparison (M1/M2/M3) across conditions."""
    conditions = ["V0", "V1", "V2", "V3"]
    models = sorted(df["model"].unique())

    for task in sorted(df["task"].unique()):
        # Best source: highest mean_metric at V3
        v3 = df[(df["task"] == task) & (df["condition"] == "V3")]
        best_source: str | None = (
            v3.groupby("source")["mean_metric"].mean().idxmax()
            if not v3.empty else None
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in models:
            pts_m, pts_s = [], []
            for cond in conditions:
                q = df[
                    (df["task"] == task) & (df["model"] == model)
                    & (df["condition"] == cond)
                ]
                if best_source is not None:
                    q_src = q[q["source"] == best_source]
                    if not q_src.empty:
                        q = q_src
                pts_m.append(float(q["mean_metric"].mean()) if not q.empty else float("nan"))
                pts_s.append(float(q["std_metric"].mean()) if not q.empty else 0.0)

            ax.errorbar(
                conditions, pts_m, yerr=pts_s,
                label=model, marker="o", capsize=4,
            )

        ax.set_xlabel("Condition")
        ax.set_ylabel(_METRIC_LABELS.get(task, "Metric"))
        title_src = f" (Source: {best_source})" if best_source else ""
        ax.set_title(f"Model Comparison — {task.upper()}{title_src}")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        _save_figure(fig, out, f"model_comparison_{task}")


def _plot_confusion_matrices(df: pd.DataFrame, out: Path) -> None:
    """Plot 5: Confusion matrix heatmaps (FC only, best source at V3)."""
    fc_df = df[df["task"] == "fc"]
    if fc_df.empty or "confusion_matrix_json" not in fc_df.columns:
        return

    for model in sorted(fc_df["model"].unique()):
        model_df = fc_df[fc_df["model"] == model]
        v3 = model_df[model_df["condition"] == "V3"]
        pool = v3 if not v3.empty else model_df
        if pool.empty:
            continue

        best_src = pool.groupby("source")["mean_metric"].mean().idxmax()
        row = pool[pool["source"] == best_src].iloc[0]

        cm_json = row.get("confusion_matrix_json")
        if not cm_json or pd.isna(cm_json):
            continue

        cm_list = json.loads(cm_json)
        cm = np.mean([np.array(c, dtype=float) for c in cm_list], axis=0)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.where(row_sums == 0, 1, row_sums)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=_FC_CLASSES, yticklabels=_FC_CLASSES,
            cmap="Blues", ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — FC {model} (V3, Source={best_src})")
        _save_figure(fig, out, f"confusion_matrix_fc_{model}")


def _plot_foe_quality(df: pd.DataFrame, out: Path) -> None:
    """Plot 6: FOE quality stratification (Bad vs Good quality)."""
    foe_df = df[df["task"] == "foe"]
    if foe_df.empty or "per_quality_json" not in foe_df.columns:
        return

    for model in sorted(foe_df["model"].unique()):
        sub = foe_df[foe_df["model"] == model].copy()
        sub = sub[sub["per_quality_json"].notna()]
        if sub.empty:
            continue

        sources, bad_means, good_means = [], [], []
        for _, row in sub.iterrows():
            try:
                quality_data: list = json.loads(row["per_quality_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            bad_vals, good_vals = [], []
            for seed_q in quality_data:
                if not isinstance(seed_q, dict):
                    continue
                if "B" in seed_q:
                    bad_vals.append(seed_q["B"].get("mean_angular_error_deg", float("nan")))
                if "G" in seed_q:
                    good_vals.append(seed_q["G"].get("mean_angular_error_deg", float("nan")))

            if not bad_vals and not good_vals:
                continue

            sources.append(row["source"])
            bad_means.append(float(np.nanmean(bad_vals)) if bad_vals else float("nan"))
            good_means.append(float(np.nanmean(good_vals)) if good_vals else float("nan"))

        if not sources:
            continue

        x = np.arange(len(sources))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x, bad_means, width, label="Bad quality")
        ax.bar(x + width, good_means, width, label="Good quality")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(sources, rotation=45, ha="right")
        ax.set_xlabel("Source")
        ax.set_ylabel("Mean Angular Error (°)")
        ax.set_title(f"FOE Quality Stratification — {model}")
        ax.legend(title="Quality", bbox_to_anchor=(1.05, 1), loc="upper left")
        _save_figure(fig, out, f"foe_quality_{model}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def generate_plots(
    results_path: str = "results/master_results.csv",
    output_dir: str = "results/figures",
) -> None:
    """Generates all paper figures and saves as .pdf and .png."""
    rp = Path(results_path)
    if not rp.exists():
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    if df.empty:
        print("No data in results file.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _apply_style()

    _plot_volume_scaling(df, out)
    _plot_substitution(df, out)
    _plot_source_comparison(df, out)
    _plot_model_comparison(df, out)
    _plot_confusion_matrices(df, out)
    _plot_foe_quality(df, out)

    print("Plot generation complete.")


if __name__ == "__main__":
    generate_plots()
