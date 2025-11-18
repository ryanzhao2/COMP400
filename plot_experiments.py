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


def ensure_out_dir(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def _pick_recall_col(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (column, label) choosing true_recall when available, else estimated recall.

    This ensures plots reflect actual accuracy when the exact baseline was computed.
    """
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
    """Compute Pareto frontier on (latency_ms, y_col).

    A point (latency, recall) is Pareto if no other point has lower latency AND higher recall.
    """
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
    """Emit additional plots using estimated recall only, for comparison/debugging."""
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


def load_experiments_from_path(log_path: str) -> pd.DataFrame:
    """Load experiments from a file or directory.
    
    If log_path is a directory, finds all .jsonl files recursively.
    If log_path is a file, loads that file.
    """
    all_records: List[Dict[str, Any]] = []
    
    if os.path.isdir(log_path):
        # Find all JSONL files in directory
        jsonl_files = []
        for root, dirs, files in os.walk(log_path):
            for file in files:
                if file.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, file))
        
        if not jsonl_files:
            print(f"No .jsonl files found in directory: {log_path}")
            return pd.DataFrame()
        
        print(f"Found {len(jsonl_files)} experiment file(s) in {log_path}")
        for jsonl_file in jsonl_files:
            print(f"  Loading: {jsonl_file}")
            df_file = load_experiments(jsonl_file)
            if not df_file.empty:
                all_records.extend(df_file.to_dict('records'))
    elif os.path.isfile(log_path):
        # Load single file
        df_file = load_experiments(log_path)
        if not df_file.empty:
            all_records.extend(df_file.to_dict('records'))
    else:
        print(f"Path does not exist: {log_path}")
        return pd.DataFrame()
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(all_records)
    return df.sort_values(["iteration"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=os.getenv("EXPERIMENT_LOG", "experiments"), 
                       help="Path to experiments JSONL file or directory containing experiment files")
    parser.add_argument("--out", default="plots", help="Output directory for plots")
    parser.add_argument("--hnsw_m", type=int, default=None, help="Clean filter: fixed hnsw_m")
    parser.add_argument("--ef_construction", type=int, default=None, help="Clean filter: fixed ef_construction")
    parser.add_argument("--database-type", dest="database_type", type=str, default=None, 
                       help="Filter by database type (knowledge_reasoning or memory_reaction)")
    parser.add_argument("--subset_name", default="fixed_params", help="Subfolder name for filtered (fixed-params) plots")
    args = parser.parse_args()

    ensure_out_dir(args.out)
    
    # Load experiments (from file or directory)
    df = load_experiments_from_path(args.log)
    if df.empty:
        print("No records found. Nothing to plot.")
        print(f"\n💡 Tip: If you're looking for experiment files, try:")
        print(f"   --log experiments/  (to load all experiment files)")
        print(f"   --log experiments/knowledge_reasoning/  (to load knowledge_reasoning experiments)")
        print(f"   --log experiments/memory_reaction/  (to load memory_reaction experiments)")
        return
    # Normalize dtypes to ensure exact-match filtering works
    for col in ["hnsw_m", "ef_construction", "ef_search", "index_size", "dataset_size", "dimension"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Filter by database type if specified
    if args.database_type:
        if "database_type" not in df.columns:
            print(f"Warning: database_type column not found in data. Cannot filter by database type.")
        else:
            available_types = sorted(df["database_type"].dropna().unique().tolist())
            df = df[df["database_type"] == args.database_type].copy()
            if df.empty:
                print(f"No records found for database_type='{args.database_type}'. Available types: {available_types}")
                return
    
    # Helpful summary
    try:
        h_vals = sorted(df["hnsw_m"].dropna().unique().tolist()) if "hnsw_m" in df.columns else []
        ec_vals = sorted(df["ef_construction"].dropna().unique().tolist()) if "ef_construction" in df.columns else []
        db_types = sorted(df["database_type"].dropna().unique().tolist()) if "database_type" in df.columns else []
        print(f"Available hnsw_m values: {h_vals}")
        print(f"Available ef_construction values: {ec_vals}")
        if db_types:
            print(f"Available database_type values: {db_types}")
    except Exception:
        pass

    # Organize plots by database type
    if "database_type" in df.columns and df["database_type"].notna().any():
        db_types = sorted(df["database_type"].dropna().unique().tolist())
        print(f"\n📊 Organizing plots by database type: {db_types}")
        
        for db_type in db_types:
            db_df = df[df["database_type"] == db_type].copy()
            if db_df.empty:
                continue
            
            # Create subfolder for this database type
            db_dir = os.path.join(args.out, db_type, "experimental")
            ensure_out_dir(db_dir)
            
            print(f"\n  Generating plots for '{db_type}' ({len(db_df)} records)...")
            plot_recall_latency(db_df, db_dir)
            plot_pareto(db_df, db_dir)
            if "ef_search" in db_df.columns and db_df["ef_search"].notna().any():
                plot_efsearch_recall(db_df, db_dir)
            plot_true_vs_estimated(db_df, db_dir)
            plot_estimated_variants(db_df, db_dir)
            
            # Clean (filtered by fixed M and EF construction), if provided
            if args.hnsw_m is not None and args.ef_construction is not None:
                clean_df = db_df[(db_df["hnsw_m"].astype("Int64") == int(args.hnsw_m)) &
                              (db_df["ef_construction"].astype("Int64") == int(args.ef_construction))].copy()
                if clean_df.empty:
                    print(f"    No records match hnsw_m={args.hnsw_m}, ef_construction={args.ef_construction} for {db_type}")
                else:
                    # Place filtered plots under fixed_params subfolder
                    clean_dir = os.path.join(args.out, db_type, args.subset_name, f"M{args.hnsw_m}_EC{args.ef_construction}")
                    ensure_out_dir(clean_dir)
                    plot_recall_latency(clean_df, clean_dir)
                    plot_pareto(clean_df, clean_dir)
                    if "ef_search" in clean_df.columns and clean_df["ef_search"].notna().any():
                        plot_efsearch_recall(clean_df, clean_dir)
                    plot_true_vs_estimated(clean_df, clean_dir)
                    plot_estimated_variants(clean_df, clean_dir)
                    print(f"    Saved filtered plots to: {clean_dir}")
            
            print(f"  ✅ Saved plots for '{db_type}' to: {os.path.abspath(db_dir)}")
    else:
        # Fallback: if no database_type, save to experimental folder (backward compatibility)
        exp_dir = os.path.join(args.out, "experimental")
        ensure_out_dir(exp_dir)
        print("\n📊 No database_type found, saving to experimental/ folder")
        plot_recall_latency(df, exp_dir)
        plot_pareto(df, exp_dir)
        if "ef_search" in df.columns and df["ef_search"].notna().any():
            plot_efsearch_recall(df, exp_dir)
        plot_true_vs_estimated(df, exp_dir)
        plot_estimated_variants(df, exp_dir)
        
        # Clean (filtered by fixed M and EF construction), if provided
        if args.hnsw_m is not None and args.ef_construction is not None:
            clean_df = df[(df["hnsw_m"].astype("Int64") == int(args.hnsw_m)) &
                          (df["ef_construction"].astype("Int64") == int(args.ef_construction))].copy()
            if clean_df.empty:
                print(f"No records match hnsw_m={args.hnsw_m}, ef_construction={args.ef_construction}. Skipping clean plots.")
                # Print top-10 most common (M, EC) pairs to help the user pick
                try:
                    freq = (df.groupby(["hnsw_m", "ef_construction"])
                              .size()
                              .reset_index(name="count")
                              .sort_values("count", ascending=False)
                              .head(10))
                    print("Most common (hnsw_m, ef_construction) pairs:")
                    for _, r in freq.iterrows():
                        print(f"  M={int(r['hnsw_m'])}, EC={int(r['ef_construction'])}: {int(r['count'])} trials")
                except Exception:
                    pass
            else:
                # Place filtered plots under a named subfolder (default: 'clean')
                clean_dir = os.path.join(args.out, args.subset_name, f"M{args.hnsw_m}_EC{args.ef_construction}")
                ensure_out_dir(clean_dir)
                plot_recall_latency(clean_df, clean_dir)
                plot_pareto(clean_df, clean_dir)
                if "ef_search" in clean_df.columns and clean_df["ef_search"].notna().any():
                    plot_efsearch_recall(clean_df, clean_dir)
                plot_true_vs_estimated(clean_df, clean_dir)
                plot_estimated_variants(clean_df, clean_dir)
        
        print(f"\n✅ Saved experimental plots to: {os.path.abspath(exp_dir)}")
        if args.hnsw_m is not None and args.ef_construction is not None:
            print(f"Filtered plots (if any) saved under: {os.path.abspath(os.path.join(args.out, args.subset_name))}")
    
    print(f"\n✅ All plots saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()


