"""
Generate comparison plots between knowledge_reasoning and memory_reaction databases.

Creates side-by-side visualizations showing recall and latency performance
for both database types, making it easy to compare their characteristics.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

def filter_experiments(df: pd.DataFrame, target_iterations: int = 5, max_experiments: int = 50) -> pd.DataFrame:
    """Filter to keep only max_experiments with exactly target_iterations."""
    filtered_dfs = []
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
            
        # Group by run_id to count iterations
        # Check if run has exactly the target iterations (0 to target_iterations-1)
        run_counts = db_df.groupby("run_id")["iteration"].nunique()
        valid_runs = run_counts[run_counts == target_iterations].index.tolist()
        

        if len(valid_runs) > max_experiments:
            valid_runs = sorted(valid_runs)[:max_experiments]
            
        # Keep only selected runs
        filtered_dfs.append(db_df[db_df["run_id"].isin(valid_runs)])
        
    if not filtered_dfs:
        return pd.DataFrame()
        
    return pd.concat(filtered_dfs)


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
            
        n_experiments = db_df["run_id"].nunique()
        iterations_per_exp = db_df.groupby("run_id")["iteration"].nunique().mode()[0] if not db_df.empty else 0
        
        plt.figure(figsize=(8, 6))
        plt.hist(recall_data, bins=20, alpha=0.7, edgecolor="black")
        plt.axvline(recall_data.mean(), color="red", linestyle="--", 
                    linewidth=2, label=f"Mean: {recall_data.mean():.3f}")
        plt.axvline(recall_data.median(), color="blue", linestyle="--", 
                   linewidth=2, label=f"Median: {recall_data.median():.3f}")
        
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{db_type.replace('_', ' ').title()} - Recall Distribution\n({n_experiments} experiments, {iterations_per_exp} iterations each)", 
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
        
        # Save to subfolder
        db_out_dir = os.path.join(out_dir, db_type)
        os.makedirs(db_out_dir, exist_ok=True)
        filename = "recall_distribution.png"
        plt.savefig(os.path.join(db_out_dir, filename), dpi=150)
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
            
        n_experiments = db_df["run_id"].nunique()
        iterations_per_exp = db_df.groupby("run_id")["iteration"].nunique().mode()[0] if not db_df.empty else 0
        
        plt.figure(figsize=(8, 6))
        plt.hist(latency_data, bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(latency_data.mean(), color="red", linestyle="--", 
                   linewidth=2, label=f"Mean: {latency_data.mean():.1f}ms")
        plt.axvline(latency_data.median(), color="blue", linestyle="--", 
                   linewidth=2, label=f"Median: {latency_data.median():.1f}ms")
        
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{db_type.replace('_', ' ').title()} - Latency Distribution\n({n_experiments} experiments, {iterations_per_exp} iterations each)", 
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
        
        # Save to subfolder
        db_out_dir = os.path.join(out_dir, db_type)
        os.makedirs(db_out_dir, exist_ok=True)
        filename = "latency_distribution.png"
        plt.savefig(os.path.join(db_out_dir, filename), dpi=150)
        plt.close()


def plot_boxplot_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Separate box plots for recall and latency for each database type."""
    recall_col = _pick_recall_col(df)
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        n_experiments = db_df["run_id"].nunique()
        iterations_per_exp = db_df.groupby("run_id")["iteration"].nunique().mode()[0] if not db_df.empty else 0
        
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
        
        plt.suptitle(f"{db_type.replace('_', ' ').title()} - Distribution Analysis\n({n_experiments} experiments, {iterations_per_exp} iterations each)", 
                    fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        # Save to subfolder
        db_out_dir = os.path.join(out_dir, db_type)
        os.makedirs(db_out_dir, exist_ok=True)
        filename = "boxplot_distribution.png"
        plt.savefig(os.path.join(db_out_dir, filename), dpi=150)
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
            
        n_experiments = db_df["run_id"].nunique()
        iterations_per_exp = db_df.groupby("run_id")["iteration"].nunique().mode()[0] if not db_df.empty else 0
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(scatter_df["latency_ms"], scatter_df[recall_col], 
                             c=scatter_df["ef_search"], cmap="viridis", 
                             s=60, alpha=0.6, edgecolors="black", linewidth=0.5)
        
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Recall" if recall_col == "true_recall" else "Recall (estimated)", fontsize=12)
        plt.title(f"{db_type.replace('_', ' ').title()} - Recall vs Latency\n({n_experiments} experiments, {iterations_per_exp} iterations each)", 
                 fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("ef_search", fontsize=11)
        
        # Set log scale for latency if needed
        if scatter_df["latency_ms"].max() / scatter_df["latency_ms"].min() > 100:
            plt.xscale("log")
        
        plt.tight_layout()
        
        # Save to subfolder
        db_out_dir = os.path.join(out_dir, db_type)
        os.makedirs(db_out_dir, exist_ok=True)
        filename = "scatter_recall_latency.png"
        plt.savefig(os.path.join(db_out_dir, filename), dpi=150)
        plt.close()


def plot_iteration_convergence(df: pd.DataFrame, out_dir: str, max_experiments: int = 50, target_iterations: int = 5, specific_iterations: Optional[List[int]] = None, filename_suffix: str = "") -> None:
    """Create convergence plots showing recall and latency over iterations.
    
    Filters to experiments with exactly target_iterations iterations,
    selects up to max_experiments such experiments, and plots their
    convergence curves for both database types.
    
    If specific_iterations is provided, only those iterations will be plotted.
    """
    recall_col = _pick_recall_col(df)
    
    # If specific_iterations is provided, use it; otherwise use target_iterations
    if specific_iterations is not None:
        iterations_to_plot = specific_iterations
        min_iterations_required = len(iterations_to_plot)
    else:
        iterations_to_plot = None
        min_iterations_required = target_iterations
    
    # First pass: collect all data to determine global y-axis limits
    all_recall_data = []
    all_latency_data = []
    valid_runs_by_db = {}
    
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        # Group by run_id to find experiments with required iterations
        run_iterations = db_df.groupby("run_id")["iteration"].apply(lambda x: sorted(x.unique().tolist())).reset_index()
        run_iterations.columns = ["run_id", "iterations_list"]
        
        # Filter to runs that have all required iterations
        if specific_iterations is not None:
            valid_runs = []
            for _, row in run_iterations.iterrows():
                if all(iter_val in row["iterations_list"] for iter_val in specific_iterations):
                    valid_runs.append(row["run_id"])
        else:
            run_iterations["unique_count"] = run_iterations["iterations_list"].apply(len)
            valid_runs = run_iterations[
                (run_iterations["unique_count"] == target_iterations)
            ]["run_id"].tolist()
        
        if len(valid_runs) == 0:
            continue
        
        # Select up to max_experiments runs
        if len(valid_runs) > max_experiments:
            valid_runs = sorted(valid_runs)[:max_experiments]
        
        valid_runs_by_db[db_type] = valid_runs
        
        # Filter dataframe to selected runs
        filtered_df = db_df[db_df["run_id"].isin(valid_runs)].copy()
        
        # If specific_iterations is provided, filter to only those iterations
        if specific_iterations is not None:
            filtered_df = filtered_df[filtered_df["iteration"].isin(iterations_to_plot)].copy()
        
        # Collect all recall and latency data for global limits
        all_recall_data.extend(filtered_df[recall_col].dropna().tolist())
        all_latency_data.extend(filtered_df["latency_ms"].dropna().tolist())
    
    # Calculate global y-axis limits
    if all_recall_data:
        recall_min = min(all_recall_data)
        recall_max = max(all_recall_data)
        recall_padding = (recall_max - recall_min) * 0.05  # 5% padding
        recall_ylim = (max(0, recall_min - recall_padding), recall_max + recall_padding)
    else:
        recall_ylim = None
    
    if all_latency_data:
        latency_min = min(all_latency_data)
        latency_max = max(all_latency_data)
        latency_padding = (latency_max - latency_min) * 0.05  # 5% padding
        latency_ylim = (max(0, latency_min - latency_padding), latency_max + latency_padding)
    else:
        latency_ylim = None
    
    # Second pass: create plots with shared scales
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        if db_type not in valid_runs_by_db:
            continue
        
        db_df = df[df["database_type"] == db_type].copy()
        if db_df.empty:
            continue
        
        valid_runs = valid_runs_by_db[db_type]
        
        if specific_iterations and len(specific_iterations) == 5:
            iter_desc = "5 iterations per experiment"
        else:
            iter_desc = f"iterations {iterations_to_plot}" if specific_iterations else f"{target_iterations} iterations"
        if len(valid_runs) > max_experiments:
            print(f"  Selected {max_experiments} out of {len(valid_runs)} experiments for {db_type}")
        else:
            print(f"  Using {len(valid_runs)} experiments (all with {iter_desc}) for {db_type}")
        
        # Filter dataframe to selected runs
        filtered_df = db_df[db_df["run_id"].isin(valid_runs)].copy()
        
        # If specific_iterations is provided, filter to only those iterations
        if specific_iterations is not None:
            filtered_df = filtered_df[filtered_df["iteration"].isin(iterations_to_plot)].copy()
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Recall over iterations
        for run_id in valid_runs:
            run_data = filtered_df[filtered_df["run_id"] == run_id].sort_values("iteration")
            # Filter to only the iterations we want to plot
            if specific_iterations is not None:
                run_data = run_data[run_data["iteration"].isin(iterations_to_plot)].sort_values("iteration")
            
            # Remove duplicates if any (keep first occurrence)
            run_data = run_data.drop_duplicates(subset=["iteration"], keep="first")
            
            if len(run_data) >= min_iterations_required:
                # Only plot if we have consecutive iterations (no gaps that would create vertical lines)
                iterations = run_data["iteration"].values
                if len(iterations) > 1 and all(iterations[i+1] - iterations[i] == 1 for i in range(len(iterations)-1)):
                    axes[0].plot(
                        run_data["iteration"], 
                        run_data[recall_col],
                        alpha=0.3,
                        linewidth=1,
                        color="blue" if db_type == "knowledge_reasoning" else "orange"
                    )
        
        # Add mean line (only for iterations we're plotting)
        mean_recall = filtered_df.groupby("iteration")[recall_col].mean()
        if specific_iterations is not None:
            mean_recall = mean_recall[mean_recall.index.isin(iterations_to_plot)].sort_index()
        
        axes[0].plot(
            mean_recall.index,
            mean_recall.values,
            linewidth=2.5,
            color="red",
            label="Mean",
            marker="o",
            markersize=6
        )
        
        # Get actual iteration values from data
        if specific_iterations is not None:
            actual_iterations = sorted([i for i in specific_iterations if i in filtered_df["iteration"].unique()])
        else:
            actual_iterations = sorted(filtered_df["iteration"].unique())
        
        axes[0].set_xlabel("Iteration", fontsize=12)
        axes[0].set_ylabel("Recall", fontsize=12)
        if specific_iterations and len(specific_iterations) == 5:
            iter_desc = "5 iterations per experiment"
        else:
            iter_desc = f"iterations {iterations_to_plot}" if specific_iterations else f"{target_iterations} iterations"
        axes[0].set_title(f"{db_type.replace('_', ' ').title()} - Recall Convergence\n({len(valid_runs)} experiments, {iter_desc})", 
                         fontsize=12, fontweight="bold")
        axes[0].set_xticks(actual_iterations)
        axes[0].grid(True, alpha=0.3)
        if recall_ylim is not None:
            axes[0].set_ylim(recall_ylim)
        axes[0].legend()
        
        # Plot 2: Latency over iterations
        for run_id in valid_runs:
            run_data = filtered_df[filtered_df["run_id"] == run_id].sort_values("iteration")
            # Filter to only the iterations we want to plot
            if specific_iterations is not None:
                run_data = run_data[run_data["iteration"].isin(iterations_to_plot)].sort_values("iteration")
            
            # Remove duplicates if any (keep first occurrence)
            run_data = run_data.drop_duplicates(subset=["iteration"], keep="first")
            
            if len(run_data) >= min_iterations_required:
                # Only plot if we have consecutive iterations (no gaps that would create vertical lines)
                iterations = run_data["iteration"].values
                if len(iterations) > 1 and all(iterations[i+1] - iterations[i] == 1 for i in range(len(iterations)-1)):
                    axes[1].plot(
                        run_data["iteration"],
                        run_data["latency_ms"],
                        alpha=0.3,
                        linewidth=1,
                        color="blue" if db_type == "knowledge_reasoning" else "orange"
                    )
        
        # Add mean line (only for iterations we're plotting)
        mean_latency = filtered_df.groupby("iteration")["latency_ms"].mean()
        if specific_iterations is not None:
            mean_latency = mean_latency[mean_latency.index.isin(iterations_to_plot)].sort_index()
        
        axes[1].plot(
            mean_latency.index,
            mean_latency.values,
            linewidth=2.5,
            color="red",
            label="Mean",
            marker="o",
            markersize=6
        )
        
        axes[1].set_xlabel("Iteration", fontsize=12)
        axes[1].set_ylabel("Latency (ms)", fontsize=12)
        if specific_iterations and len(specific_iterations) == 5:
            iter_desc = "5 iterations per experiment"
        else:
            iter_desc = f"iterations {iterations_to_plot}" if specific_iterations else f"{target_iterations} iterations"
        axes[1].set_title(f"{db_type.replace('_', ' ').title()} - Latency Convergence\n({len(valid_runs)} experiments, {iter_desc})", 
                         fontsize=12, fontweight="bold")
        axes[1].set_xticks(actual_iterations)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("linear")  
        if latency_ylim is not None:
            axes[1].set_ylim(latency_ylim)
        axes[1].legend()
        
        plt.tight_layout()
        if filename_suffix:
            filename = f"convergence_{filename_suffix}.png"
        else:
            filename = f"convergence.png"
            
        db_out_dir = os.path.join(out_dir, db_type)
        os.makedirs(db_out_dir, exist_ok=True)
        plt.savefig(os.path.join(db_out_dir, filename), dpi=150, bbox_inches="tight")
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
    
    # Save to comparison subfolder
    comp_dir = os.path.join(out_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    plt.savefig(os.path.join(comp_dir, "statistics_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots between database types")
    parser.add_argument("--experiments", default="experiments", 
                       help="Path to experiments directory")
    parser.add_argument("--out", default="plots", 
                       help="Output directory for comparison plots")
    parser.add_argument("--dataset-size", type=int, default=1000000,
                       help="Filter by dataset size (default: 1000000)")
    parser.add_argument("--max-experiments", type=int, default=50,
                       help="Maximum number of experiments to show in convergence plots (default: 50)")
    parser.add_argument("--target-iterations", type=int, default=5,
                       help="Target number of iterations per experiment (default: 5)")
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
    
    # Filter to specific number of experiments with specific iterations
    print(f"Filtering to {args.max_experiments} experiments with {args.target_iterations} iterations each...")
    df = filter_experiments(df, args.target_iterations, args.max_experiments)
    
    if df.empty:
        print("No experiments matched the filtering criteria.")
        return

    print(f"Working with {len(df)} total records after filtering")
    for db_type in ["knowledge_reasoning", "memory_reaction"]:
        db_df = df[df["database_type"] == db_type]
        if not db_df.empty:
            print(f"  {db_type}: {db_df['run_id'].nunique()} experiments ({len(db_df)} records)")
    
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
    
    # Create convergence plots - reuse the filtered dataframe
    # We pass None for specific_iterations to rely on the already filtered data's iterations
    # But plot_iteration_convergence expects specific_iterations for the x-axis or logic
    # We can pass the range(target_iterations)
    plot_iteration_convergence(df, args.out, max_experiments=args.max_experiments, 
                             specific_iterations=list(range(args.target_iterations)), 
                             target_iterations=args.target_iterations)
    print(f"  Created iteration convergence plots for iterations {list(range(args.target_iterations))} for each database type")
    
    print(f"\nAll comparison plots saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

