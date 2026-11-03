from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: E402
from scipy.stats import wilcoxon

from src.analysis.plots import _apply_style, _save_figure, _METRIC_LABELS
from src.analysis.tables import to_latex

# fc/fmd: higher mean_metric is better. foe: mean_metric is an error in
# degrees, lower is better -- delta is normalised per-task below so that
# "positive delta" always means "per-condition tuning helped", regardless
# of which task it is, so every downstream aggregate/plot/test in this file
# can be read the same way across fc/fmd/foe without a case split.
_HIGHER_IS_BETTER = {"fc": True, "fmd": True, "foe": False}

_CONDITION_TYPE = {
    "V0": "baseline", "R0": "baseline",
    "V1": "volume", "V2": "volume", "V3": "volume",
    "R1": "substitution", "R2": "substitution",
    "R3": "substitution", "R4": "substitution",
}


def build_comparison(primary_path: str, fixedhp_path: str) -> pd.DataFrame:
    """
    Joins the primary (per-condition-tuned) and Fixed-HP results on
    (task, model, condition, source) -- both CSVs carry these as plain
    columns already (not parsed from experiment_id), and every Fixed-HP
    experiment has an exact-scope counterpart in the primary batch (same
    693 non-baseline + 9 baseline grid, per generate_fixedhp_configs.py),
    so this is a clean 1:1 join, not an approximate match.
    """
    primary = pd.read_csv(primary_path)
    fixedhp = pd.read_csv(fixedhp_path)

    key = ["task", "model", "condition", "source"]
    merged = primary[key + ["mean_metric", "std_metric"]].merge(
        fixedhp[key + ["mean_metric", "std_metric"]],
        on=key, suffixes=("_tuned", "_fixed"), how="inner",
    )

    missing_primary = len(fixedhp) - len(merged)
    if missing_primary:
        print(f"WARNING: {missing_primary} Fixed-HP rows had no primary-batch "
              f"match on {key} -- check both CSVs cover the same grid.")

    def _delta(row) -> float:
        d = row["mean_metric_tuned"] - row["mean_metric_fixed"]
        return d if _HIGHER_IS_BETTER[row["task"]] else -d

    merged["delta"] = merged.apply(_delta, axis=1)
    merged["condition_type"] = merged["condition"].map(_CONDITION_TYPE)
    return merged


def summarise(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Per (task, model): mean/std delta, fraction of experiments where
    per-condition tuning wins, and a paired one-sided Wilcoxon signed-rank
    test (H1: delta > 0, i.e. tuning helps) -- the actual significance
    evidence for the causal claim, not just an eyeballed mean.
    """
    rows = []
    for (task, model), g in merged.groupby(["task", "model"]):
        deltas = g["delta"].to_numpy()
        n = len(deltas)
        win_frac = float((deltas > 0).mean())
        if np.allclose(deltas, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(deltas, alternative="greater")
        rows.append({
            "task": task, "model": model, "n": n,
            "mean_delta": deltas.mean(), "std_delta": deltas.std(),
            "win_frac": win_frac, "wilcoxon_stat": stat, "p_greater": p,
        })
    return pd.DataFrame(rows).sort_values(["task", "model"])


def summarise_by_condition_type(merged: pd.DataFrame) -> pd.DataFrame:
    """Same as summarise(), but split by volume/substitution/baseline --
    the two regimes may show a different-sized (or different-signed)
    tuning benefit, worth reporting separately rather than only pooled."""
    rows = []
    for (task, model, ctype), g in merged.groupby(
        ["task", "model", "condition_type"]
    ):
        deltas = g["delta"].to_numpy()
        n = len(deltas)
        win_frac = float((deltas > 0).mean())
        if n < 2 or np.allclose(deltas, 0):
            stat, p = float("nan"), float("nan")
        else:
            stat, p = wilcoxon(deltas, alternative="greater")
        rows.append({
            "task": task, "model": model, "condition_type": ctype, "n": n,
            "mean_delta": deltas.mean(), "std_delta": deltas.std(),
            "win_frac": win_frac, "p_greater": p,
        })
    return pd.DataFrame(rows).sort_values(["task", "model", "condition_type"])


def save_tables(summary: pd.DataFrame, by_ctype: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    disp = summary.copy()
    disp["mean_delta"] = disp["mean_delta"].map(lambda x: f"{x:.4f}")
    disp["std_delta"] = disp["std_delta"].map(lambda x: f"{x:.4f}")
    disp["win_frac"] = disp["win_frac"].map(lambda x: f"{x:.1%}")
    disp["p_greater"] = disp["p_greater"].map(lambda x: f"{x:.4g}")
    disp = disp.drop(columns=["wilcoxon_stat"]).set_index(["task", "model"])
    (out / "fixedhp_comparison_summary.csv").write_text(disp.to_csv())
    (out / "fixedhp_comparison_summary.tex").write_text(to_latex(
        disp,
        caption="Per-condition tuning vs. Fixed-HP: mean delta (positive = "
                "tuning helps, task-normalised), win fraction, and one-sided "
                "Wilcoxon signed-rank p-value.",
        label="tab:fixedhp:summary",
    ))

    disp2 = by_ctype.copy()
    disp2["mean_delta"] = disp2["mean_delta"].map(lambda x: f"{x:.4f}")
    disp2["std_delta"] = disp2["std_delta"].map(lambda x: f"{x:.4f}")
    disp2["win_frac"] = disp2["win_frac"].map(lambda x: f"{x:.1%}")
    disp2["p_greater"] = disp2["p_greater"].map(
        lambda x: f"{x:.4g}" if pd.notna(x) else "n/a"
    )
    disp2 = disp2.set_index(["task", "model", "condition_type"])
    (out / "fixedhp_comparison_by_condition_type.csv").write_text(disp2.to_csv())
    (out / "fixedhp_comparison_by_condition_type.tex").write_text(to_latex(
        disp2,
        caption="Per-condition tuning vs. Fixed-HP, split by condition type.",
        label="tab:fixedhp:by_condition_type",
    ))

    print(f"  Saved fixedhp_comparison_summary.{{csv,tex}}, "
          f"fixedhp_comparison_by_condition_type.{{csv,tex}}")


def plot_delta_distribution(merged: pd.DataFrame, out: Path) -> None:
    """One panel per task: delta distribution per model, zero-line
    reference. Positive = per-condition tuning helped."""
    _apply_style()
    tasks = sorted(merged["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = merged[merged["task"] == task]
        sns.boxplot(data=sub, x="model", y="delta", ax=ax)
        sns.stripplot(data=sub, x="model", y="delta", ax=ax,
                       color="black", alpha=0.35, size=3)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{task.upper()} — {_METRIC_LABELS.get(task, task)}")
        ax.set_ylabel("Δ (tuning benefit, task-normalised)")
        ax.set_xlabel("Model")
    fig.suptitle("Per-condition tuning vs. Fixed-HP")
    _save_figure(fig, out, "fixedhp_delta_distribution")


def plot_scatter(merged: pd.DataFrame, out: Path) -> None:
    """Tuned vs Fixed-HP raw mean_metric, one panel per task, y=x reference
    line -- points above the line (fc/fmd) or below it (foe, since lower is
    better there) indicate tuning helped for that specific experiment."""
    _apply_style()
    tasks = sorted(merged["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = merged[merged["task"] == task]
        sns.scatterplot(
            data=sub, x="mean_metric_fixed", y="mean_metric_tuned",
            hue="model", style="condition_type", ax=ax, alpha=0.7,
        )
        lo = min(sub["mean_metric_fixed"].min(), sub["mean_metric_tuned"].min())
        hi = max(sub["mean_metric_fixed"].max(), sub["mean_metric_tuned"].max())
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{task.upper()} — {_METRIC_LABELS.get(task, task)}")
        ax.set_xlabel("Fixed-HP")
        ax.set_ylabel("Per-condition tuned")
    fig.suptitle("Tuned vs. Fixed-HP (points above/below the diagonal favor "
                  "tuning, direction depends on task)")
    _save_figure(fig, out, "fixedhp_scatter")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-condition-tuned vs Fixed-HP results."
    )
    parser.add_argument("--primary", default="results/master_results.csv")
    parser.add_argument("--fixedhp", default="results/master_results_fixedhp.csv")
    parser.add_argument("--tables-out", default="results/tables")
    parser.add_argument("--figures-out", default="results/figures")
    args = parser.parse_args()

    merged = build_comparison(args.primary, args.fixedhp)
    print(f"Joined {len(merged)} paired (task,model,condition,source) rows.")

    summary = summarise(merged)
    by_ctype = summarise_by_condition_type(merged)
    print(summary.to_string(index=False))

    save_tables(summary, by_ctype, Path(args.tables_out))

    figout = Path(args.figures_out)
    figout.mkdir(parents=True, exist_ok=True)
    plot_delta_distribution(merged, figout)
    plot_scatter(merged, figout)
    print("Fixed-HP comparison complete.")


if __name__ == "__main__":
    main()
