import json
import networkx as nx
import torch
import numpy as np
from typing import List, Tuple
from torch_geometric.data import Data as PyGData

def to_valid_array(input_value):
    if isinstance(input_value, list):
        return input_value

def load_graph_from_json(data, dataset_score_attribute):

    q = data
    
    adj = to_valid_array(q["dep_graph"])
    num_nodes = len(adj)
    G = nx.DiGraph()

    nodes = [f"Node_{i}" for i in range(num_nodes)]
    G.add_nodes_from(nodes)

    for node_id in nodes:
        G.nodes[node_id]['score'] = q["claims"][int(node_id.split("_")[1])][dataset_score_attribute] + q["claims"][int(node_id.split("_")[1])]["noise"] 

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j]:
                G.add_edge(nodes[i], nodes[j])
                    
    
    print(f"--- loaded one graph ---")
    return G


def generate_growth_subgraphs(
    G: nx.DiGraph,
    use_prefix: bool = True,
    max_steps: int = None,
) -> List[nx.DiGraph]:

    if G.number_of_nodes() == 0:
        print("generate_growth_subgraphs: empty graph")
        return []

    order = list(nx.topological_sort(G)) 

    if max_steps is not None:
        order = order[:max_steps]

    subgraphs = []
    empty_subgraph = G.subgraph([]).copy()
    subgraphs.append(empty_subgraph)
    
    kept = []

    for v in order:
        kept.append(v)
        subG = G.subgraph(kept).copy()
        subgraphs.append(subG)

    return subgraphs

def nx_to_pyg_data_fixed(subG: nx.DiGraph) -> "PyGData":

    if PyGData is None:
        raise RuntimeError("PyTorch Geometric not available. Please install torch-geometric.")

    nodes = list(subG.nodes())
    node2idx = {u: i for i, u in enumerate(nodes)}

    x_list = []
    for u in nodes:
        s = float(subG.nodes[u].get("score", 0.0))
        indeg = float(subG.in_degree(u))
        outdeg = float(subG.out_degree(u))
        ann = subG.nodes[u].get("annotation", None)
        ann_val = 1.0 if ann == "Y" else (0.0 if ann == "N" else -1.0)
        x_list.append([s, indeg, outdeg, ann_val])
    x = torch.tensor(x_list, dtype=torch.float32)

    edges = list(subG.edges())
    if not edges:
        edge_index = torch.arange(len(nodes), dtype=torch.long).unsqueeze(0).repeat(2, 1)
    else:
        edge_index = torch.tensor([[node2idx[u], node2idx[v]] for (u, v) in edges], dtype=torch.long).t().contiguous()

    return PyGData(x=x, edge_index=edge_index)

def normalized_laplacian_eigvals_from_edge_index(edge_index: np.ndarray, n: int, k: int) -> np.ndarray:

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
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = 1.0 / np.sqrt(deg)
    inv_sqrt_deg[~np.isfinite(inv_sqrt_deg)] = 0.0
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    I = np.eye(n, dtype=np.float32)
    L = I - (D_inv_sqrt @ A @ D_inv_sqrt)

    evals = np.linalg.eigvalsh(L).astype(np.float32)
    evals.sort()

    if evals.shape[0] >= k:
        return evals[:k]
    out = np.full((k,), 2.0, dtype=np.float32)
    out[: evals.shape[0]] = evals
    return out

def pyg_to_features(pyg_data, k_eigs: int = 8) -> np.ndarray:

    n = int(pyg_data.x.shape[0]) if hasattr(pyg_data, "x") and pyg_data.x is not None else 0

    if hasattr(pyg_data, "edge_index") and pyg_data.edge_index is not None:
        ei = pyg_data.edge_index.detach().cpu().numpy()
    else:
        ei = np.zeros((2, 0), dtype=np.int64)

    lap_eigs = normalized_laplacian_eigvals_from_edge_index(ei, n=n, k=k_eigs)

    if hasattr(pyg_data, "x") and pyg_data.x is not None and pyg_data.x.shape[0] > 0:
        node_scores = pyg_data.x[:, 0].cpu().numpy() 

        node_feat_stats = np.array([
            float(node_scores.mean()),
            float(node_scores.std()),
            float(node_scores.min()),
            float(node_scores.max()),
        ], dtype=np.float32)
    else:
        node_feat_stats = np.zeros((4,), dtype=np.float32)
    
    feats = np.concatenate([lap_eigs, node_feat_stats], axis=0)

    return feats

def feature_fn(subG: nx.DiGraph, k_eigs: int = 8) -> np.ndarray:

    pyg = nx_to_pyg_data_fixed(subG)           
    feats = pyg_to_features(pyg, k_eigs=k_eigs)
    feats = np.asarray(feats, dtype=np.float32)
    return feats

def risk_scores_for_sequence(
    subgraphs: List[nx.DiGraph],
    focal_model,
    beta: float = 0.25,
    max_nodes_cap: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:

    if not subgraphs:
        return np.array([]), np.array([])

    X = np.stack([feature_fn(sg) for sg in subgraphs], axis=0)  

    smx = focal_model.predict_proba(X)[:, 0]
    size_terms = np.array([len(sg.nodes())-1 for sg in subgraphs])

    score = smx + beta * size_terms 

    score[0] = -np.inf 

    return smx, score