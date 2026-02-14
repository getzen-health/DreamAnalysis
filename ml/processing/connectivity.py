"""Brain Network Connectivity Analysis Module.

Computes directed and undirected connectivity between EEG channels
including Granger causality, Directed Transfer Function (DTF),
and graph-theoretic metrics (clustering, path length, small-world index).
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, List, Tuple, Optional


def compute_granger_causality(
    signals: np.ndarray, fs: float = 256.0, max_lag: int = 10
) -> Dict:
    """Compute pairwise Granger causality between channels.

    Uses vector autoregressive model residuals to determine if one
    channel's past helps predict another channel's future.

    Args:
        signals: 2D array (n_channels, n_samples).
        fs: Sampling frequency.
        max_lag: Maximum lag in samples for the AR model.

    Returns:
        Dict with 'matrix' (n_channels x n_channels directed connectivity),
        'significant_pairs' (pairs with high GC values).
    """
    n_channels, n_samples = signals.shape
    gc_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            gc_matrix[i, j] = _pairwise_granger(signals[j], signals[i], max_lag)

    # Normalize to 0-1 range
    max_gc = np.max(gc_matrix)
    if max_gc > 0:
        gc_matrix = gc_matrix / max_gc

    # Find significant pairs (above mean + 1 std)
    threshold = np.mean(gc_matrix[gc_matrix > 0]) + np.std(gc_matrix[gc_matrix > 0]) if np.any(gc_matrix > 0) else 0
    significant = []
    for i in range(n_channels):
        for j in range(n_channels):
            if gc_matrix[i, j] > threshold:
                significant.append({
                    "from": int(i),
                    "to": int(j),
                    "strength": float(gc_matrix[i, j]),
                })

    return {
        "matrix": gc_matrix.tolist(),
        "significant_pairs": significant,
    }


def _pairwise_granger(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """Compute Granger causality from x to y (does x help predict y?)."""
    n = len(y)
    if n <= max_lag + 1:
        return 0.0

    # Restricted model: y predicted from its own past
    Y = y[max_lag:]
    X_restricted = np.column_stack([y[max_lag - k - 1 : n - k - 1] for k in range(max_lag)])

    # Unrestricted model: y predicted from its own past + x's past
    X_unrestricted = np.column_stack([
        X_restricted,
        *[x[max_lag - k - 1 : n - k - 1].reshape(-1, 1) for k in range(max_lag)],
    ])

    try:
        # Fit via least squares
        _, res_r, _, _ = np.linalg.lstsq(X_restricted, Y, rcond=None)
        _, res_u, _, _ = np.linalg.lstsq(X_unrestricted, Y, rcond=None)

        rss_r = float(res_r[0]) if len(res_r) > 0 else float(np.sum((Y - X_restricted @ np.linalg.lstsq(X_restricted, Y, rcond=None)[0]) ** 2))
        rss_u = float(res_u[0]) if len(res_u) > 0 else float(np.sum((Y - X_unrestricted @ np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]) ** 2))

        if rss_u <= 0:
            return 0.0
        return max(0.0, float(np.log(rss_r / max(rss_u, 1e-10))))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_dtf(
    signals: np.ndarray, fs: float = 256.0, freq_range: Tuple[float, float] = (1.0, 40.0)
) -> Dict:
    """Compute Directed Transfer Function via multivariate AR model.

    DTF measures the causal influence of one channel on another in
    the frequency domain. Uses Yule-Walker equations for AR fitting.

    Returns:
        Dict with 'dtf_matrix', 'dominant_direction'.
    """
    n_channels, n_samples = signals.shape
    order = min(10, n_samples // (n_channels * 3))

    # Fit multivariate AR model via Yule-Walker
    try:
        AR_coeffs = _fit_mvar(signals, order)
    except Exception:
        return {
            "dtf_matrix": np.zeros((n_channels, n_channels)).tolist(),
            "dominant_direction": "undetermined",
        }

    # Compute DTF in frequency domain
    n_freqs = 64
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    dtf_matrix = np.zeros((n_channels, n_channels))

    for f_idx, freq in enumerate(freqs):
        z = np.exp(-2j * np.pi * freq / fs)
        H = np.eye(n_channels, dtype=complex)
        for k in range(order):
            H -= AR_coeffs[k] * (z ** (-(k + 1)))
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue

        # DTF: normalized transfer function
        for j in range(n_channels):
            col_norm = np.sqrt(np.sum(np.abs(H_inv[:, j]) ** 2))
            if col_norm > 0:
                for i in range(n_channels):
                    dtf_matrix[i, j] += (np.abs(H_inv[i, j]) ** 2) / (col_norm ** 2)

    dtf_matrix /= max(n_freqs, 1)

    # Determine dominant direction
    front_to_back = np.mean(dtf_matrix[:n_channels // 2, n_channels // 2:])
    back_to_front = np.mean(dtf_matrix[n_channels // 2:, :n_channels // 2])

    if front_to_back > back_to_front * 1.2:
        direction = "frontal_to_parietal"
    elif back_to_front > front_to_back * 1.2:
        direction = "parietal_to_frontal"
    else:
        direction = "bidirectional"

    return {
        "dtf_matrix": dtf_matrix.tolist(),
        "dominant_direction": direction,
    }


def _fit_mvar(signals: np.ndarray, order: int) -> List[np.ndarray]:
    """Fit Multivariate AutoRegressive model via Yule-Walker equations."""
    n_channels, n_samples = signals.shape

    # Compute autocorrelation matrices
    R = []
    for lag in range(order + 1):
        if lag == 0:
            R.append(signals @ signals.T / n_samples)
        else:
            R.append(signals[:, lag:] @ signals[:, :-lag].T / (n_samples - lag))

    # Solve Yule-Walker equations (block Toeplitz system)
    # Simplified: solve for each lag independently
    coeffs = []
    for k in range(order):
        try:
            A_k = np.linalg.solve(R[0] + np.eye(n_channels) * 1e-6, R[k + 1])
            coeffs.append(A_k)
        except np.linalg.LinAlgError:
            coeffs.append(np.zeros((n_channels, n_channels)))

    return coeffs


def compute_graph_metrics(connectivity_matrix: np.ndarray) -> Dict:
    """Compute graph-theoretic metrics from a connectivity matrix.

    Metrics computed using numpy (no NetworkX dependency):
    - Clustering coefficient
    - Average path length
    - Small-world index
    - Hub nodes (highest degree centrality)
    - Modularity estimate

    Args:
        connectivity_matrix: 2D array (n_nodes x n_nodes), values 0-1.

    Returns:
        Dict with all graph metrics.
    """
    n = connectivity_matrix.shape[0]
    if n < 2:
        return {
            "clustering_coefficient": 0.0,
            "avg_path_length": 0.0,
            "small_world_index": 0.0,
            "hub_nodes": [],
            "modularity": 0.0,
        }

    # Threshold to create binary adjacency matrix
    threshold = np.mean(connectivity_matrix[connectivity_matrix > 0]) if np.any(connectivity_matrix > 0) else 0.5
    adj = (connectivity_matrix > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    # Degree centrality
    degrees = adj.sum(axis=1)
    max_degree = n - 1
    degree_centrality = degrees / max(max_degree, 1)

    # Hub nodes (top quartile by degree)
    hub_threshold = np.percentile(degree_centrality, 75) if n > 1 else 0
    hub_nodes = [int(i) for i in range(n) if degree_centrality[i] >= hub_threshold and degrees[i] > 0]

    # Clustering coefficient
    clustering = _compute_clustering(adj, n)

    # Average path length (via BFS on binary graph)
    avg_path = _compute_avg_path_length(adj, n)

    # Small-world index: compare to random graph
    # Random graph: C_rand ≈ k/n, L_rand ≈ ln(n)/ln(k)
    k = np.mean(degrees)
    if k > 1 and n > 1:
        c_rand = k / n
        l_rand = np.log(n) / np.log(max(k, 1.01))
        gamma = clustering / max(c_rand, 1e-10)
        lamda = avg_path / max(l_rand, 1e-10)
        small_world = gamma / max(lamda, 1e-10)
    else:
        small_world = 0.0

    # Modularity estimate (simple 2-community split)
    modularity = _estimate_modularity(adj, degrees, n)

    return {
        "clustering_coefficient": float(clustering),
        "avg_path_length": float(avg_path),
        "small_world_index": float(min(small_world, 10.0)),
        "hub_nodes": hub_nodes,
        "modularity": float(modularity),
        "degree_centrality": [float(d) for d in degree_centrality],
    }


def _compute_clustering(adj: np.ndarray, n: int) -> float:
    """Compute global clustering coefficient."""
    total_triangles = 0
    total_triples = 0

    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        total_triples += k * (k - 1) / 2
        # Count triangles
        for ni_idx in range(len(neighbors)):
            for nj_idx in range(ni_idx + 1, len(neighbors)):
                if adj[neighbors[ni_idx], neighbors[nj_idx]] > 0:
                    total_triangles += 1

    return float(total_triangles / max(total_triples, 1))


def _compute_avg_path_length(adj: np.ndarray, n: int) -> float:
    """Compute average shortest path length via BFS."""
    total_dist = 0
    total_pairs = 0

    for src in range(n):
        visited = np.full(n, -1)
        visited[src] = 0
        queue = [src]
        head = 0

        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in range(n):
                if adj[node, neighbor] > 0 and visited[neighbor] < 0:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)

        reachable = visited[visited > 0]
        total_dist += np.sum(reachable)
        total_pairs += len(reachable)

    return float(total_dist / max(total_pairs, 1))


def _estimate_modularity(adj: np.ndarray, degrees: np.ndarray, n: int) -> float:
    """Estimate modularity using spectral bisection."""
    m = np.sum(adj) / 2
    if m == 0:
        return 0.0

    # Modularity matrix
    B = adj - np.outer(degrees, degrees) / (2 * m)

    # Use leading eigenvector for partition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        leading = eigenvectors[:, -1]
        partition = (leading >= 0).astype(int)
    except np.linalg.LinAlgError:
        partition = np.zeros(n, dtype=int)
        partition[n // 2:] = 1

    # Compute modularity for this partition
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if partition[i] == partition[j]:
                Q += adj[i, j] - degrees[i] * degrees[j] / (2 * m)
    Q /= (2 * m)

    return float(max(0.0, min(1.0, Q)))


def compute_connectivity_dynamics(
    signals: np.ndarray,
    fs: float = 256.0,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> List[Dict]:
    """Compute sliding-window connectivity for temporal evolution.

    Returns:
        List of dicts, one per window, with connectivity matrix and metrics.
    """
    n_channels, n_samples = signals.shape
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    results = []

    for start in range(0, n_samples - window_samples + 1, step_samples):
        window = signals[:, start : start + window_samples]

        # Use coherence for fast windowed connectivity
        from processing.eeg_processor import compute_coherence
        coh = compute_coherence(window, fs, "alpha")

        # Simple pairwise correlation as connectivity proxy
        corr = np.corrcoef(window)
        np.fill_diagonal(corr, 0)
        corr = np.abs(corr)

        metrics = compute_graph_metrics(corr)

        results.append({
            "time": float(start / fs),
            "coherence": float(coh),
            "clustering": metrics["clustering_coefficient"],
            "path_length": metrics["avg_path_length"],
            "small_world": metrics["small_world_index"],
        })

    return results
