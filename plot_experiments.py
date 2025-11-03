"""
Read experiments.jsonl and generate plots:
 - recall vs latency
 - Pareto frontier (recall vs latency, colored by memory)
 - ef_search vs recall
 - true vs estimated recall (if true_recall present)

Usage:
  python plot_experiments.py --log experiments.jsonl --out plots
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiments(log_path: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ds = obj.get("dataset", {})
            params = obj.get("params", {})
            metrics = obj.get("metrics", {})
            rec: Dict[str, Any] = {
                "run_id": obj.get("run_id"),
                "iteration": obj.get("iteration"),
                "datetime": obj.get("datetime"),
                "dataset_size": ds.get("dataset_size"),
                "dimension": ds.get("dimension"),
                "hnsw_m": params.get("hnsw_m"),
                "ef_construction": params.get("ef_construction"),
                "ef_search": params.get("ef_search"),
                "recall": metrics.get("recall"),
                "true_recall": metrics.get("true_recall"),
                "latency_ms": metrics.get("latency_ms"),
                "memory_gb": metrics.get("memory_gb"),
                "index_size": metrics.get("index_size"),
                "k": metrics.get("k"),
            }
            records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    return df.sort_values(["iteration"]).reset_index(drop=True)


def ensure_out_dir(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _pick_recall_col(df: pd.DataFrame) -> Tuple[str, str]:
    has_true = "true_recall" in df.columns and df["true_recall"].notna().any()
    if has_true:
        return "true_recall", "True recall"
    return "recall", "Recall (estimated)"


def plot_recall_latency(df: pd.DataFrame, out_dir: str) -> None:
    y_col, y_label = _pick_recall_col(df)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="latency_ms", y=y_col, hue="ef_search", palette="viridis", s=60)
    plt.title(f"{y_label} vs Latency (colored by ef_search)")
    plt.xlabel("Latency (ms)")
    plt.ylabel(y_label)
    plt.legend(title="ef_search", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recall_vs_latency.png"), dpi=150)
    plt.close()


def pareto_frontier(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    # Higher recall is better; lower latency is better.
    pts = df.dropna(subset=[y_col, "latency_ms"]).copy()
    pts = pts.sort_values(["latency_ms", y_col], ascending=[True, False])
    pareto: List[int] = []
    best_recall = -1.0
    for idx, row in pts.iterrows():
        if row[y_col] > best_recall:
            pareto.append(idx)
            best_recall = row[y_col]
    return pts.loc[pareto]


def plot_pareto(df: pd.DataFrame, out_dir: str) -> None:
    y_col, y_label = _pick_recall_col(df)
    front = pareto_frontier(df, y_col)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="latency_ms", y=y_col, hue="memory_gb", palette="coolwarm", s=50, alpha=0.6)
    sns.lineplot(data=front.sort_values("latency_ms"), x="latency_ms", y=y_col, color="black", marker="o", label="Pareto")
    plt.title(f"Pareto Frontier: {y_label} vs Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_frontier.png"), dpi=150)
    plt.close()


def plot_efsearch_recall(df: pd.DataFrame, out_dir: str) -> None:
    y_col, y_label = _pick_recall_col(df)
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=df.sort_values("ef_search"), x="ef_search", y=y_col, marker="o")
    plt.title(f"ef_search vs {y_label}")
    plt.xlabel("ef_search")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "efsearch_vs_recall.png"), dpi=150)
    plt.close()


def plot_true_vs_estimated(df: pd.DataFrame, out_dir: str) -> None:
    sub = df.dropna(subset=["true_recall", "recall"]).copy()
    if sub.empty:
        return
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=sub, x="recall", y="true_recall", s=60)
    lims = [0, 1]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("True vs Estimated Recall")
    plt.xlabel("Estimated recall")
    plt.ylabel("True recall")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "true_vs_estimated_recall.png"), dpi=150)
    plt.close()


def plot_estimated_variants(df: pd.DataFrame, out_dir: str) -> None:
    # Only if estimated recall exists
    if "recall" not in df.columns or not df["recall"].notna().any():
        return
    # Recall vs latency (estimated only)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="latency_ms", y="recall", hue="ef_search", palette="viridis", s=60)
    plt.title("Recall (estimated) vs Latency (colored by ef_search)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Recall (estimated)")
    plt.legend(title="ef_search", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recall_vs_latency_estimated.png"), dpi=150)
    plt.close()

    # Pareto (estimated)
    pts = df.dropna(subset=["recall", "latency_ms"]).copy()
    pts = pts.sort_values(["latency_ms", "recall"], ascending=[True, False])
    pareto_idx: List[int] = []
    best = -1.0
    for idx, row in pts.iterrows():
        if row["recall"] > best:
            pareto_idx.append(idx)
            best = row["recall"]
    front = pts.loc[pareto_idx]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="latency_ms", y="recall", hue="memory_gb", palette="coolwarm", s=50, alpha=0.6)
    sns.lineplot(data=front.sort_values("latency_ms"), x="latency_ms", y="recall", color="black", marker="o", label="Pareto")
    plt.title("Pareto Frontier: Recall (estimated) vs Latency")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Recall (estimated)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto_frontier_estimated.png"), dpi=150)
    plt.close()

    # ef_search vs estimated recall
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=df.sort_values("ef_search"), x="ef_search", y="recall", marker="o")
    plt.title("ef_search vs Recall (estimated)")
    plt.xlabel("ef_search")
    plt.ylabel("Recall (estimated)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "efsearch_vs_recall_estimated.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=os.getenv("EXPERIMENT_LOG", "experiments.jsonl"), help="Path to experiments JSONL")
    parser.add_argument("--out", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    ensure_out_dir(args.out)
    if not os.path.isfile(args.log):
        print(f"No log file at {args.log}. Nothing to plot.")
        return

    df = load_experiments(args.log)
    if df.empty:
        print("No records found in log. Nothing to plot.")
        return

    # Basic plots
    plot_recall_latency(df, args.out)
    plot_pareto(df, args.out)
    if "ef_search" in df.columns and df["ef_search"].notna().any():
        plot_efsearch_recall(df, args.out)
    plot_true_vs_estimated(df, args.out)
    # Also emit estimated-only variants for comparison
    plot_estimated_variants(df, args.out)

    print(f"Saved plots to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()


