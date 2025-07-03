#!/usr/bin/env python3
"""
summarize_inversion_and_plot.py

Aggregates mean L₂ inversion errors from DP vs. raw for each of your four variants
(gaussian_normal, gaussian_malicious, laplace_normal, laplace_malicious) and
produces:

1) A console table
2) inversion_error_summary.csv
3) inversion_error_comparison.png (bar chart with exact values on top)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Use this script’s directory as the base
    base_dir = os.path.dirname(os.path.abspath(__file__))

    variants = [
        "gaussian_normal",
        "gaussian_malicous",
        "laplace_normal",
        "laplace_malicious",
    ]

    summary = []
    for var in variants:
        dp_file  = os.path.join(base_dir, var, "inversion_results_dp_round25.xlsx")
        raw_file = os.path.join(base_dir, var, "inversion_results_raw_round25.xlsx")

        if not os.path.exists(dp_file):
            print(f"[!] Missing file: {dp_file}")
            continue
        if not os.path.exists(raw_file):
            print(f"[!] Missing file: {raw_file}")
            continue

        # Read the "errors" sheet
        df_dp  = pd.read_excel(dp_file,  sheet_name="errors", header=0)
        df_raw = pd.read_excel(raw_file, sheet_name="errors", header=0)

        # Only the first 10 data rows
        df_dp  = df_dp.iloc[:10]
        df_raw = df_raw.iloc[:10]

        # Convert to numeric and drop invalids
        dp_errors  = pd.to_numeric(df_dp["L2_error"], errors="coerce").dropna()
        raw_errors = pd.to_numeric(df_raw["L2_error"], errors="coerce").dropna()

        # Compute means
        mean_dp  = dp_errors.mean()
        mean_raw = raw_errors.mean()

        summary.append({
            "Variant": var.replace("_", " ").title(),
            "Mean L2 (DP)":  round(mean_dp, 4),
            "Mean L2 (Raw)": round(mean_raw, 4)
        })

    if not summary:
        print("No data processed. Check your subfolders and filenames.")
        return

    # Build summary DataFrame
    df_summary = pd.DataFrame(summary)

    # 1) Print to console
    print("\nModel-Inversion Error Summary:\n")
    print(df_summary.to_string(index=False))

    # 2) Save as CSV
    out_csv = os.path.join(base_dir, "inversion_error_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV to: {out_csv}")

    # 3) Plot a bar chart comparing DP vs Raw
    df_plot = df_summary.set_index("Variant")[["Mean L2 (DP)", "Mean L2 (Raw)"]]
    ax = df_plot.plot(
        kind="bar",
        figsize=(10, 6),
        rot=45,
        title="Inversion Attack: DP vs Raw Mean L₂ Error"
    )
    ax.set_ylabel("Mean L₂ Error")

    # Annotate exact values on top of each bar
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),           # 4 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    # Save and show
    out_png = os.path.join(base_dir, "inversion_error_comparison.png")
    plt.savefig(out_png)
    print(f"Saved comparison plot to: {out_png}")
    plt.show()

if __name__ == "__main__":
    main()
