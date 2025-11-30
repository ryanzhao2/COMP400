"""
Generate comparison plots between knowledge_reasoning and memory_reaction databases.

Creates side-by-side visualizations showing recall and latency performance
for both database types, making it easy to compare their characteristics.
"""

import os
import json
import argparse
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# Use the same loading function from plot_experiments.py
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
                "database_type": obj.get("database_type") or ds.get("database_type") or "knowledge_reasoning",
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


def load_all_experiments(experiments_dir: str = "experiments") -> pd.DataFrame:
    """Load all experiments from both database types."""
    all_records: List[Dict[str, Any]] = []
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_dir = os.path.join(experiments_dir, db_type)
        if not os.path.isdir(db_dir):
            continue
        
        for filename in os.listdir(db_dir):
            if filename.startswith("experiments_n") and filename.endswith(".jsonl"):
                filepath = os.path.join(db_dir, filename)
                df = load_experiments(filepath)
                if not df.empty:
                    all_records.extend(df.to_dict('records'))
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(all_records)
    # Normalize dtypes
    for col in ["hnsw_m", "ef_construction", "ef_search", "index_size", "dataset_size", "dimension"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def _pick_recall_col(df: pd.DataFrame) -> str:
    """Choose true_recall if available, else recall."""
    if "true_recall" in df.columns and df["true_recall"].notna().any():
        return "true_recall"
    return "recall"


def plot_recall_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Separate recall plots for each database type."""
    recall_col = _pick_recall_col(df)
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        recall_data = db_df[recall_col].dropna()
        if len(recall_data) == 0:
            continue
        
        plt.figure(figsize=(8, 6))
        plt.hist(recall_data, bins=20, alpha=0.7, edgecolor="black")
        plt.axvline(recall_data.mean(), color="red", linestyle="--", 
                    linewidth=2, label=f"Mean: {recall_data.mean():.3f}")
        plt.axvline(recall_data.median(), color="blue", linestyle="--", 
                   linewidth=2, label=f"Median: {recall_data.median():.3f}")
        
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{db_type.replace('_', ' ').title()} - Recall Distribution\n(n={len(recall_data)} experiments)", 
                 fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Set y-axis ticks to show more values (every integer from 0 to max)
        max_freq = int(plt.gca().get_ylim()[1]) + 1
        if max_freq <= 20:
            # For small ranges, show every integer
            plt.yticks(range(0, max_freq))
        else:
            # For larger ranges, show every 2-5 values
            step = max(1, max_freq // 15)
            plt.yticks(range(0, max_freq, step))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        plt.tight_layout()
        
        filename = f"recall_{db_type}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()


def plot_latency_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Separate latency plots for each database type."""
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        latency_data = db_df["latency_ms"].dropna()
        if len(latency_data) == 0:
            continue
        
        plt.figure(figsize=(8, 6))
        plt.hist(latency_data, bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(latency_data.mean(), color="red", linestyle="--", 
                   linewidth=2, label=f"Mean: {latency_data.mean():.1f}ms")
        plt.axvline(latency_data.median(), color="blue", linestyle="--", 
                   linewidth=2, label=f"Median: {latency_data.median():.1f}ms")
        
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{db_type.replace('_', ' ').title()} - Latency Distribution\n(n={len(latency_data)} experiments)", 
                 fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Use linear scale (not log) for unbiased frequency axis
        # Set y-axis ticks to show more values
        max_freq = int(plt.gca().get_ylim()[1]) + 1
        if max_freq <= 20:
            # For small ranges, show every integer
            plt.yticks(range(0, max_freq))
        else:
            # For larger ranges, show reasonable intervals
            step = max(1, max_freq // 15)
            plt.yticks(range(0, max_freq, step))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        plt.tight_layout()
        
        filename = f"latency_{db_type}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()


def plot_boxplot_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Separate box plots for recall and latency for each database type."""
    recall_col = _pick_recall_col(df)
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Recall box plot
        recall_data = db_df[recall_col].dropna()
        if len(recall_data) > 0:
            axes[0].boxplot([recall_data], tick_labels=[db_type.replace("_", "\n")])
            axes[0].set_ylabel("Recall", fontsize=12)
            axes[0].set_title("Recall Distribution", fontsize=12, fontweight="bold")
            axes[0].grid(True, alpha=0.3, axis="y")
        
        # Latency box plot
        latency_data = db_df["latency_ms"].dropna()
        if len(latency_data) > 0:
            axes[1].boxplot([latency_data], tick_labels=[db_type.replace("_", "\n")])
            axes[1].set_ylabel("Latency (ms)", fontsize=12)
            axes[1].set_title("Latency Distribution", fontsize=12, fontweight="bold")
            axes[1].set_yscale("log")
            axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.suptitle(f"{db_type.replace('_', ' ').title()} - Distribution Analysis", 
                    fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        filename = f"boxplot_{db_type}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()


def plot_scatter_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Separate scatter plots showing recall vs latency for each database type."""
    recall_col = _pick_recall_col(df)
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        scatter_df = db_df.dropna(subset=[recall_col, "latency_ms"])
        if len(scatter_df) == 0:
            continue
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(scatter_df["latency_ms"], scatter_df[recall_col], 
                             c=scatter_df["ef_search"], cmap="viridis", 
                             s=60, alpha=0.6, edgecolors="black", linewidth=0.5)
        
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Recall" if recall_col == "true_recall" else "Recall (estimated)", fontsize=12)
        plt.title(f"{db_type.replace('_', ' ').title()} - Recall vs Latency\n(n={len(scatter_df)} experiments)", 
                 fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("ef_search", fontsize=11)
        
        # Set log scale for latency if needed
        if scatter_df["latency_ms"].max() / scatter_df["latency_ms"].min() > 100:
            plt.xscale("log")
        
        plt.tight_layout()
        
        filename = f"scatter_{db_type}.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()


def plot_statistics_summary(df: pd.DataFrame, out_dir: str) -> None:
    """Create a clean summary table with statistics for both database types."""
    recall_col = _pick_recall_col(df)
    
    stats = []
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        recall_data = db_df[recall_col].dropna()
        latency_data = db_df["latency_ms"].dropna()
        
        if len(recall_data) > 0 and len(latency_data) > 0:
            stats.append({
                "Metric": "Recall",
                "Mean": f"{recall_data.mean():.3f}",
                "Std Dev": f"{recall_data.std():.3f}",
                "Min": f"{recall_data.min():.3f}",
                "Max": f"{recall_data.max():.3f}",
            })
            stats.append({
                "Metric": "Latency (ms)",
                "Mean": f"{latency_data.mean():.2f}",
                "Std Dev": f"{latency_data.std():.2f}",
                "Min": f"{latency_data.min():.2f}",
                "Max": f"{latency_data.max():.2f}",
            })
    
    if not stats:
        return
    
    # Create separate tables for each database type
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    for idx, db_type in enumerate(["knowledge_reasoning", "memory_reaction"]):
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        recall_data = db_df[recall_col].dropna()
        latency_data = db_df["latency_ms"].dropna()
        
        if len(recall_data) > 0 and len(latency_data) > 0:
            table_data = [
                ["Recall", f"{recall_data.mean():.3f}", f"{recall_data.std():.3f}", 
                 f"{recall_data.min():.3f}", f"{recall_data.max():.3f}"],
                ["Latency (ms)", f"{latency_data.mean():.2f}", f"{latency_data.std():.2f}", 
                 f"{latency_data.min():.2f}", f"{latency_data.max():.2f}"],
            ]
            
            axes[idx].axis("tight")
            axes[idx].axis("off")
            
            table = axes[idx].table(
                cellText=table_data,
                colLabels=["Metric", "Mean", "Std Dev", "Min", "Max"],
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor("#2E7D32")
                table[(0, i)].set_text_props(weight="bold", color="white")
            
            # Style data rows
            for i in range(1, 3):
                for j in range(5):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor("#F5F5F5")
                    else:
                        table[(i, j)].set_facecolor("#FFFFFF")
                    table[(i, j)].set_text_props(size=10)
            
            # Set title
            title = db_type.replace("_", " ").title()
            axes[idx].set_title(f"{title}\n(n={len(db_df)} experiments)", 
                              fontsize=12, fontweight="bold", pad=15)
    
    plt.suptitle("Performance Statistics Summary", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "statistics_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots between database types")
    parser.add_argument("--experiments", default="experiments", 
                       help="Path to experiments directory")
    parser.add_argument("--out", default="plots/comparison", 
                       help="Output directory for comparison plots")
    parser.add_argument("--dataset-size", type=int, default=1000000,
                       help="Filter by dataset size (default: 1000000)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load all experiments
    print("Loading experiments...")
    df = load_all_experiments(args.experiments)
    
    if df.empty:
        print("No experiment data found. Make sure experiments are in the experiments/ directory.")
        return
    
    # Filter by dataset size
    if "dataset_size" in df.columns:
        df = df[df["dataset_size"] == args.dataset_size].copy()
        print(f"Filtered to dataset_size = {args.dataset_size}")
    
    if df.empty:
        print(f"No experiments found with dataset_size = {args.dataset_size}")
        return
    
    print(f"Loaded {len(df)} total experiments")
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type]
        if not db_df.empty:
            print(f"  {db_type}: {len(db_df)} experiments")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_recall_comparison(df, args.out)
    print("  Created recall plots for each database type")
    
    plot_latency_comparison(df, args.out)
    print("  Created latency plots for each database type")
    
    plot_boxplot_comparison(df, args.out)
    print("  Created boxplot plots for each database type")
    
    plot_scatter_comparison(df, args.out)
    print("  Created scatter plots for each database type")
    
    plot_statistics_summary(df, args.out)
    print("  Created statistics_summary.png")
    
    print(f"\nAll comparison plots saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

