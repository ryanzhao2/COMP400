"""
Utilities for generating and loading synthetic vector datasets and queries.

Provides several distributions:
- Gaussian clusters
- Uniform on a hypersphere
- Power-law radial distribution
"""

from typing import Tuple
import os
import math
import numpy as np


def set_seed(seed: int) -> None:
    # Set numpy RNG seed (kept for potential future use)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)


def generate_gaussian_clusters(num_vectors: int, dimension: int, num_clusters: int = 8, cluster_std: float = 0.1, seed: int = 42) -> np.ndarray:
    # Generate vectors clustered around random Gaussian centers
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(num_clusters, dimension)).astype(np.float32)
    counts = np.full(num_clusters, num_vectors // num_clusters, dtype=int)
    counts[: num_vectors % num_clusters] += 1
    chunks = []
    for i in range(num_clusters):
        noise = rng.normal(0.0, cluster_std, size=(counts[i], dimension)).astype(np.float32)
        chunks.append(centers[i] + noise)
    return np.vstack(chunks).astype(np.float32)


def generate_uniform_sphere(num_vectors: int, dimension: int, seed: int = 42) -> np.ndarray:
    # Generate unit vectors uniformly distributed on the hypersphere
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(num_vectors, dimension)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype(np.float32)


def generate_powerlaw(num_vectors: int, dimension: int, alpha: float = 2.0, seed: int = 42) -> np.ndarray:
    # Vectors with uniform directions and power-law radii p(r) ~ r^{-alpha} on (0, 1]
    rng = np.random.default_rng(seed)
    # Directions
    dirs = rng.normal(0.0, 1.0, size=(num_vectors, dimension)).astype(np.float32)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    # Radii via inverse CDF for alpha > 1 over (0,1]
    u = rng.random(num_vectors).astype(np.float32)
    # r = u^{1/(1-alpha)}
    r = np.power(u, 1.0 / (1.0 - alpha)).astype(np.float32)
    return (dirs * r[:, None]).astype(np.float32)


def generate_queries(vectors: np.ndarray, num_queries: int = 100, seed: int = 123) -> np.ndarray:
    # Create queries by sampling dataset points and adding small Gaussian noise
    rng = np.random.default_rng(seed)
    n, d = vectors.shape
    if n == 0:
        raise ValueError("Empty dataset provided for query generation")
    idx = rng.choice(n, size=min(num_queries, n), replace=False)
    # Slightly perturb selected dataset points to create queries near data manifold
    queries = vectors[idx].copy()
    noise = rng.normal(0.0, 0.01, size=queries.shape).astype(np.float32)
    return (queries + noise).astype(np.float32)


def save_npy(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def summarize(array: np.ndarray) -> Tuple[int, int, float, float]:
    n, d = array.shape
    mn = float(np.min(array))
    mx = float(np.max(array))
    return n, d, mn, mx


