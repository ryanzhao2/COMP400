"""
CLI to generate multiple synthetic vector datasets and queries.

Usage:
  python generate_datasets.py --out data --n 10000 --d 128 --clusters 8 --std 0.1
"""

import os
import argparse
from typing import Tuple
import numpy as np

from datasets import (
    generate_gaussian_clusters,
    generate_uniform_sphere,
    generate_powerlaw,
    generate_queries,
    save_npy,
)


def write_set(name: str, vectors: np.ndarray, out_dir: str, num_queries: int = 100) -> Tuple[str, str]:
    base = os.path.join(out_dir, name)
    vec_path = base + ".npy"
    q_path = base + "_queries.npy"
    save_npy(vec_path, vectors)
    queries = generate_queries(vectors, num_queries=num_queries)
    save_npy(q_path, queries)
    return vec_path, q_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument("--n", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--d", type=int, default=128, help="Vector dimension")
    parser.add_argument("--clusters", type=int, default=8, help="Number of Gaussian clusters")
    parser.add_argument("--std", type=float, default=0.1, help="Stddev for Gaussian clusters")
    parser.add_argument("--alpha", type=float, default=2.0, help="Power-law alpha (>1)")
    parser.add_argument("--queries", type=int, default=200, help="Number of queries to generate")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Gaussian clusters
    g = generate_gaussian_clusters(args.n, args.d, num_clusters=args.clusters, cluster_std=args.std)
    g_vec, g_q = write_set(f"gaussian_n{args.n}_d{args.d}_k{args.clusters}_std{args.std}", g, args.out, args.queries)
    print(f"Gaussian: vectors={g_vec}, queries={g_q}")

    # Uniform sphere
    s = generate_uniform_sphere(args.n, args.d)
    s_vec, s_q = write_set(f"sphere_n{args.n}_d{args.d}", s, args.out, args.queries)
    print(f"Sphere:   vectors={s_vec}, queries={s_q}")

    # Power-law
    p = generate_powerlaw(args.n, args.d, alpha=args.alpha)
    p_vec, p_q = write_set(f"powerlaw_n{args.n}_d{args.d}_a{args.alpha}", p, args.out, args.queries)
    print(f"Powerlaw: vectors={p_vec}, queries={p_q}")


if __name__ == "__main__":
    main()


