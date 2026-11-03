from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# LaTeX helper
# ---------------------------------------------------------------------------

def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format a DataFrame as a booktabs LaTeX table with caption and label."""
    # pandas 2.x removed booktabs kwarg from DataFrame.to_latex;
    # emulate it by replacing hline markers.
    latex_body = df.to_latex(escape=False)
    latex_body = (
        latex_body
        .replace(r"\hline", "", 1)           # remove first \hline after \begin{tabular}
        .replace(r"\hline", r"\bottomrule", 1)  # last \hline → \bottomrule
    )
    # Insert toprule / midrule after \begin{tabular}{...}
    lines = latex_body.splitlines()
    out_lines: list[str] = []
    header_done = False
    for line in lines:
        out_lines.append(line)
        if line.startswith(r"\begin{tabular}"):
            out_lines.append(r"\toprule")
        elif not header_done and line.strip() and not line.startswith(r"\begin") \
                and not line.startswith(r"\toprule"):
            # first data row (header) — add midrule after it
            out_lines.append(r"\midrule")
            header_done = True
    latex_body = "\n".join(out_lines)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex_body}"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------

def _metric_str(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _pivot_mean_std(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
) -> pd.DataFrame:
    """
    Build a pivot table where each cell is "mean ± std".
    Aggregates by mean when multiple rows match the same (index, column) pair.
    """
    index_vals = sorted(df[index_col].dropna().unique())
    col_vals   = sorted(df[columns_col].dropna().unique())

    data: dict[str, list] = {c: [] for c in col_vals}
    for idx in index_vals:
        for col in col_vals:
            subset = df[(df[index_col] == idx) & (df[columns_col] == col)]
            if subset.empty:
                data[col].append("-")
            else:
                m = float(subset["mean_metric"].mean())
                s = float(subset["std_metric"].mean())
                data[col].append(_metric_str(m, s))

    return pd.DataFrame(data, index=index_vals)


def _save_table(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    caption: str,
    label: str,
) -> None:
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    df.to_csv(csv_path)
    tex_path.write_text(to_latex(df, caption, label), encoding="utf-8")
    print(f"  Saved {csv_path.name}, {tex_path.name}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_tables(
    results_path: str = "results/master_results.csv",
    output_dir: str = "results/tables",
) -> None:
    """Generates all paper tables as .csv and .tex files."""
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

    tasks  = sorted(df["task"].unique())
    models = sorted(df["model"].unique())

    # ── Table 1: Volume scaling (V0–V3) per task per model ─────────────────
    for task in tasks:
        for model in models:
            sub = df[
                (df["task"] == task) & (df["model"] == model)
                & df["condition"].str.startswith("V")
            ]
            if sub.empty:
                continue
            table = _pivot_mean_std(sub, "condition", "source")
            _save_table(
                table, out,
                name=f"table1_volume_{task}_{model}",
                caption=f"Volume scaling — task={task}, model={model}",
                label=f"tab:volume:{task}:{model}",
            )

    # ── Table 2: Substitution (R0–R4) per task per model ───────────────────
    for task in tasks:
        for model in models:
            sub = df[
                (df["task"] == task) & (df["model"] == model)
                & df["condition"].str.startswith("R")
            ]
            if sub.empty:
                continue
            table = _pivot_mean_std(sub, "condition", "source")
            _save_table(
                table, out,
                name=f"table2_substitution_{task}_{model}",
                caption=f"Substitution — task={task}, model={model}",
                label=f"tab:sub:{task}:{model}",
            )

    # ── Table 3: Model comparison (M1 vs M2 vs M3) ─────────────────────────
    for task in tasks:
        sub = df[(df["task"] == task) & df["condition"].str.startswith("V")]
        if sub.empty:
            continue
        table = _pivot_mean_std(sub, "model", "condition")
        _save_table(
            table, out,
            name=f"table3_models_{task}",
            caption=f"Model comparison — task={task}",
            label=f"tab:models:{task}",
        )

    # ── Table 4: Source ranking at V3 ──────────────────────────────────────
    for model in models:
        sub_v3 = df[(df["condition"] == "V3") & (df["model"] == model)]
        if sub_v3.empty:
            continue

        rows = []
        for src in sorted(sub_v3["source"].unique()):
            row: dict = {"source": src}
            for task in tasks:
                subset = sub_v3[
                    (sub_v3["task"] == task) & (sub_v3["source"] == src)
                ]
                if subset.empty:
                    row[task] = "-"
                else:
                    m = float(subset["mean_metric"].mean())
                    s = float(subset["std_metric"].mean())
                    row[task] = _metric_str(m, s)
            rows.append(row)

        if not rows:
            continue

        table = pd.DataFrame(rows).set_index("source")

        # Sort by first task (best first); handle "-" gracefully
        def _sort_key(val: str) -> float:
            try:
                return float(val.split(" ")[0])
            except (ValueError, AttributeError):
                return float("-inf")

        if tasks:
            table = table.iloc[
                table[tasks[0]].map(_sort_key).argsort()[::-1].values
            ]

        _save_table(
            table, out,
            name=f"table4_ranking_{model}",
            caption=f"Source ranking at V3 — model={model}",
            label=f"tab:rank:{model}",
        )

    print("Table generation complete.")


if __name__ == "__main__":
    generate_tables()
