import numpy as np

def normalized_laplacian_eigvals_from_edge_index(edge_index: np.ndarray, n: int, k: int) -> np.ndarray:
    """
    Compute the smallest-k eigenvalues of the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    using undirected adjacency A (symmetrized).

    edge_index: shape (2, E) numpy array, directed edges allowed.
    n: number of nodes
    k: number of eigenvalues to return
    """
    if n <= 0:
        return np.full((k,), 2.0, dtype=np.float32)

    # Build undirected adjacency matrix (binary)
    A = np.zeros((n, n), dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0].astype(int)
        dst = edge_index[1].astype(int)
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # symmetrize

    # Degrees
    deg = A.sum(axis=1)
    # Handle isolated nodes safely
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = 1.0 / np.sqrt(deg)
    inv_sqrt_deg[~np.isfinite(inv_sqrt_deg)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    # Normalized Laplacian
    I = np.eye(n, dtype=np.float32)
    L = I - (D_inv_sqrt @ A @ D_inv_sqrt)

    # Eigenvalues (symmetric PSD; numerical noise possible)
    evals = np.linalg.eigvalsh(L).astype(np.float32)
    evals.sort()

    # Take smallest k; if n<k pad with 2.0 (max possible eigenvalue for normalized Laplacian)
    if evals.shape[0] >= k:
        return evals[:k]
    out = np.full((k,), 2.0, dtype=np.float32)
    out[: evals.shape[0]] = evals
    return out

def structural_stats_from_edge_index(edge_index: np.ndarray, n: int) -> np.ndarray:
    """
    Basic structural stats using directed edges (as provided).
    Returns a small vector of floats.
    """
    E = int(edge_index.shape[1]) if edge_index.size > 0 else 0
    if n <= 0:
        return np.zeros((10,), dtype=np.float32)

    # in/out degrees in directed sense
    indeg = np.zeros((n,), dtype=np.float32)
    outdeg = np.zeros((n,), dtype=np.float32)
    if E > 0:
        src = edge_index[0].astype(int)
        dst = edge_index[1].astype(int)
        np.add.at(outdeg, src, 1.0)
        np.add.at(indeg, dst, 1.0)

    n_float = float(n)
    density = E / (n_float * (n_float - 1.0) + 1e-12)  # directed density (no self-loops)

    sources = float((indeg == 0).sum()) / n_float
    sinks = float((outdeg == 0).sum()) / n_float

    stats = np.array([
        n_float,
        float(E),
        float(density),
        float(indeg.mean()),
        float(outdeg.mean()),
        float(indeg.max()) if n > 0 else 0.0,
        float(outdeg.max()) if n > 0 else 0.0,
        float(indeg.var()),
        float(outdeg.var()),
        float(sources + sinks),  # simple “DAG-ish” indicator; optional
    ], dtype=np.float32)
    return stats


def pyg_to_features(pyg_data, k_eigs: int = 8) -> np.ndarray:
    """
    Build feature vector = [k Laplacian eigenvalues + structural stats]
    """
    # number of nodes
    n = int(pyg_data.x.shape[0]) if hasattr(pyg_data, "x") and pyg_data.x is not None else 0

    # edge_index
    if hasattr(pyg_data, "edge_index") and pyg_data.edge_index is not None:
        ei = pyg_data.edge_index.detach().cpu().numpy()
    else:
        ei = np.zeros((2, 0), dtype=np.int64)

    lap_eigs = normalized_laplacian_eigvals_from_edge_index(ei, n=n, k=k_eigs)
    stats = structural_stats_from_edge_index(ei, n=n)

    # add node feature statistics
    if hasattr(pyg_data, "x") and pyg_data.x is not None and pyg_data.x.shape[0] > 0:
        node_scores = pyg_data.x[:, 0].cpu().numpy()

        # Min-Max normalization to [0, 1]
        score_min = node_scores.min()
        score_max = node_scores.max()
        if score_max > score_min:  
            node_scores = (node_scores - score_min) / (score_max - score_min)
        else:
            node_scores = node_scores 

        node_feat_stats = np.array([
            float(node_scores.mean()),
            float(node_scores.std()),
            float(node_scores.min()),
            float(node_scores.max()),
        ], dtype=np.float32)
    else:
        node_feat_stats = np.zeros((4,), dtype=np.float32)
    
    # feats = np.concatenate([lap_eigs, stats, node_feat_stats], axis=0)
    feats = np.concatenate([lap_eigs, node_feat_stats], axis=0)

    # feats = np.concatenate([lap_eigs, stats], axis=0).astype(np.float32)
    # feats = np.concatenate([lap_eigs], axis=0).astype(np.float32)
    return feats