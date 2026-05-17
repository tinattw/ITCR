import json
import random
from typing import List, Any, Optional, Dict, Tuple

import numpy as np
import networkx as nx
import torch

try:
    from torch_geometric.data import Data as PyGData
except Exception:
    PyGData = None

import pickle

def save_graph_list(graphs: List[nx.DiGraph], path: str):
    with open(path, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_graph_list(path: str) -> List[nx.DiGraph]:
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return graphs

# makes dependency graph an array if not already
def to_valid_array(input_value):
    # Check if the input is already a list
    if isinstance(input_value, list):
        return input_value

def load_graphs_from_json(json_path, dataset, dataset_score_attribute, annotation_attribute):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data.get("data", [])
    
    all_graphs = []
    for idx, q in enumerate(questions):
        adj = to_valid_array(q["dep_graph"])
        num_nodes = len(adj)
        G = nx.DiGraph(name=f"{idx}")

        nodes = [f"Node_{i}" for i in range(num_nodes)]
        G.add_nodes_from(nodes)

        for node_id in nodes:
            G.nodes[node_id]['score'] = q["claims"][int(node_id.split("_")[1])][dataset_score_attribute] + q["claims"][int(node_id.split("_")[1])]["noise"]
            G.nodes[node_id]['annotation'] = q["claims"][int(node_id.split("_")[1])][annotation_attribute]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i][j]:
                    if dataset == "felm_wk" or dataset == "gsm8k":
                        G.add_edge(nodes[i], nodes[j])
                    else:
                        G.add_edge(nodes[j], nodes[i])
                    
        all_graphs.append(G)
    print(f"--- loaded {len(all_graphs)} graphs ---")
    return all_graphs

def split_graphs(
    all_graphs: List[nx.DiGraph],
    seed: int = 42,
    train_ratio: float = 0.3,
    return_indices: bool = False,
) -> Tuple:
    """
    Split graphs into train / calib / test sets to avoid subgraph leakage.

    Parameters:
      - all_graphs: List[nx.DiGraph]
      - seed: random seed
      - train_ratio / calib_ratio / test_ratio
      - return_indices: True to return the original index list for each split

    Returns:
      - return_indices=False: (train_graphs, calib_graphs, test_graphs)
      - return_indices=True: (train_graphs, calib_graphs, test_graphs, train_idx, calib_idx, test_idx)
    """
    if not all_graphs: raise ValueError("all_graphs is empty")
    if train_ratio < 0: raise ValueError("Ratios must be non-negative")

    n = len(all_graphs)
    idx = np.arange(n)

    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    calib_test_ratio = 1 - train_ratio

    n_train = int(np.floor(n * train_ratio))
    n_calib_test = int(np.floor(n * calib_test_ratio))

    train_idx = idx[:n_train]
    calib_test_idx = idx[n_train:n_train + n_calib_test]
    train_graphs = [all_graphs[i] for i in train_idx]
    calib_test_graphs = [all_graphs[i] for i in calib_test_idx]

    if return_indices:
        return train_graphs, calib_test_graphs, train_idx.tolist(), calib_test_idx.tolist()
    return train_graphs, calib_test_graphs



def nx_to_pyg_data_fixed(subG: nx.DiGraph) -> "PyGData":
    """
    Convert NetworkX DiGraph subgraph to PyG Data.
    Node features: [score, indeg, outdeg, ann_val]
      - score: subG.nodes[u]["score"] (float)
      - indeg/outdeg: subgraph inner degree
      - ann_val: "Y"->1, "N"->0, missing->-1
    """
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
        # use self-loop to avoid empty edge_index
        edge_index = torch.arange(len(nodes), dtype=torch.long).unsqueeze(0).repeat(2, 1)
    else:
        edge_index = torch.tensor([[node2idx[u], node2idx[v]] for (u, v) in edges], dtype=torch.long).t().contiguous()

    return PyGData(x=x, edge_index=edge_index)


def random_walk_nodes(
    base_graph: nx.Graph,
    start_node: Any,
    target_size: int,
    walk_factor: int = 3,
) -> List[Any]:
    """
    Random walk on base_graph.
    """
    visited = [start_node]
    cur = start_node
    max_steps = max(1, target_size * walk_factor)

    for _ in range(max_steps):
        nbrs = list(base_graph.neighbors(cur))
        if not nbrs:
            break
        cur = random.choice(nbrs)
        if cur not in visited:
            visited.append(cur)
        if len(visited) >= target_size:
            break

    return visited


def closure_with_ancestors(G: nx.DiGraph, nodes: List[Any]) -> List[Any]:
    """
    Ancestor closure: if the subgraph contains v, then all ancestors of v are also included, ensuring more complete dependency semantics.
    """
    keep = set(nodes)
    for v in list(nodes):
        keep.update(nx.ancestors(G, v))
    return list(keep)


def compute_subgraph_label_ratio(subG: nx.DiGraph, pos_token: str = "Y") -> float:
    """
    Binary label:
      1.0 = completely correct subgraph (all nodes annotation == pos_token)
      0.0 = incomplete correct subgraph (exists any node annotation != pos_token)
    """
    nodes = list(subG.nodes())
    if not nodes:
        return 1.0 # empty subgraph is considered coherent-correct
    for _, data in subG.nodes(data=True):
        if data.get("annotation", None) != pos_token:
            return 0.0 # if any node is not Y, it is considered incomplete correct
    return 1.0

def compute_subgraph_label_ratio_fraction(subG: nx.DiGraph, pos_token: str = "Y") -> float:
    """
    Subgraph label = (#nodes annotation == pos_token) / (#total nodes in subgraph)
    Return continuous ratio value ∈ [0,1]
    """
    nodes = list(subG.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0

    pos = 0
    for _, data in subG.nodes(data=True):
        if str(data.get("annotation", "")).strip().upper() == pos_token:
            pos += 1

    return pos / n


def build_subgraph_dataset_rw_binary(
    graphs: List[nx.DiGraph],
    k_candidates: int,
    seed: int,
    min_nodes: int = 1,
    max_nodes_frac: float = 0.9,
    max_nodes_cap: int = 64,
    walk_factor: int = 3,
    use_undirected_walk: bool = True,
    do_ancestor_closure: bool = True,
    balance_start_nodes: bool = True,
    pos_token: str = "Y",
    neg_token: str = "N",
    truncate_after_closure: bool = False,
    deduplicate: bool = True,
    verbose: bool = True,
) -> List["PyGData"]:
    
    if PyGData is None:
        raise RuntimeError("PyTorch Geometric not available. Please install torch-geometric.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_samples: List[PyGData] = []
    total_before_dedup = 0
    pos_count = 0
    node_sizes = []
    edge_sizes = []

    for gid, G in enumerate(graphs):
        n = G.number_of_nodes()
        if n == 0:
            continue

        max_nodes = min(max(min_nodes, int(np.ceil(n * max_nodes_frac))), max_nodes_cap)

        # select base graph for random walk
        if use_undirected_walk:
            base_graph = G.to_undirected()
        else:
            base_graph = G.to_undirected()

        all_nodes = list(G.nodes())

        # start node pool: optional balance
        if balance_start_nodes:
            pos_nodes = [u for u, d in G.nodes(data=True) if d.get("annotation", None) == pos_token]
            neg_nodes = [u for u, d in G.nodes(data=True) if d.get("annotation", None) == neg_token]
            # if one type is empty, fall back to all nodes
            if not pos_nodes:
                pos_nodes = all_nodes
            if not neg_nodes:
                neg_nodes = all_nodes
        else:
            pos_nodes, neg_nodes = all_nodes, all_nodes

        # generate candidate subgraphs
        subGs: List[nx.DiGraph] = []
        for sid in range(k_candidates):
            target_size = random.randint(min_nodes, max_nodes)

            if balance_start_nodes:
                # alternate sampling from positive/negative start nodes
                start_pool = pos_nodes if (sid % 2 == 0) else neg_nodes
            else:
                start_pool = all_nodes

            start = random.choice(start_pool)
            visited = random_walk_nodes(base_graph, start, target_size, walk_factor=walk_factor)

            kept = visited
            if do_ancestor_closure:
                kept = closure_with_ancestors(G, kept)

            # after closure, it may explode, so truncate
            if truncate_after_closure and len(kept) > max_nodes:
                keep_set = set(visited)
                # add ancestors until max_nodes
                for v in visited:
                    for a in nx.ancestors(G, v):
                        keep_set.add(a)
                        if len(keep_set) >= max_nodes:
                            break
                    if len(keep_set) >= max_nodes:
                        break
                kept = list(keep_set)
                if len(kept) > max_nodes:
                    kept = kept[:max_nodes]

            if len(kept) < min_nodes:
                continue

            subG = G.subgraph(kept).copy()
            if subG.number_of_nodes() >= min_nodes:
                subGs.append(subG)

        # deduplicate (by node set)
        if deduplicate:
            uniq: Dict[Tuple[Any, ...], nx.DiGraph] = {}
            for sg in subGs:
                key = tuple(sorted(list(sg.nodes())))
                uniq[key] = sg
            subGs = list(uniq.values())

        total_before_dedup += k_candidates

        # convert to PyG + label
        for sid, subG in enumerate(subGs):
            y = compute_subgraph_label_ratio(subG, pos_token=pos_token)
            # ratio = compute_subgraph_label_ratio_fraction(subG) 
            pyg = nx_to_pyg_data_fixed(subG)
            # pyg.y_ratio = torch.tensor([ratio], dtype=torch.float32)
            pyg.y = torch.tensor([y], dtype=torch.float32)

            pyg.graph_id = torch.tensor([gid], dtype=torch.long)
            pyg.subgraph_id = torch.tensor([sid], dtype=torch.long)

            all_samples.append(pyg)

            pos_count += int(y == 1.0)
            node_sizes.append(subG.number_of_nodes())
            edge_sizes.append(subG.number_of_edges())

    if verbose:
        total = len(all_samples)
        pos_rate = (pos_count / total) if total > 0 else 0.0
        uniq_rate = (total / total_before_dedup) if total_before_dedup > 0 else 0.0

        def _safe_stats(arr):
            if not arr:
                return (0, 0, 0)
            return (int(np.min(arr)), float(np.mean(arr)), int(np.max(arr)))

        nmin, nmean, nmax = _safe_stats(node_sizes)
        emin, emean, emax = _safe_stats(edge_sizes)

        print(f"[build_subgraph_dataset_rw_binary]")
        print(f"  graphs: {len(graphs)}")
        print(f"  requested subgraphs: {total_before_dedup}")
        print(f"  kept subgraphs: {total} (unique_rate~{uniq_rate:.3f})")
        print(f"  pos_count: {pos_count}, pos_rate: {pos_rate:.4f}")
        print(f"  nodes per subgraph: min={nmin}, mean={nmean:.2f}, max={nmax}")
        print(f"  edges per subgraph: min={emin}, mean={emean:.2f}, max={emax}")

    return all_samples

if __name__ == "__main__":

    dataset = 'gsm8k' 
    json_path = f'data/{dataset}_data.json'
    dataset_score_attribute = "frequency-score"
    annotation_attribute = "annotation"

    all_graphs = load_graphs_from_json(json_path, dataset, dataset_score_attribute, annotation_attribute)
    train_graphs, calib_test_graphs = split_graphs(all_graphs, seed=42)
    save_graph_list(calib_test_graphs, f"data/pre-trained/{dataset}/calib_test_graphs_{dataset}.pkl")
    save_graph_list(train_graphs, f"data/pre-trained/{dataset}/train_graphs_{dataset}.pkl")
    
    train_samples = build_subgraph_dataset_rw_binary(
        train_graphs, k_candidates=16, seed=42
    )
    calib_test_samples = build_subgraph_dataset_rw_binary(
        calib_test_graphs, k_candidates=16, seed=43
    )
    torch.save(train_samples, f"data/pre-trained/{dataset}/train_pyg_{dataset}.pt")
    torch.save(calib_test_samples, f"data/pre-trained/{dataset}/calib_test_pyg_{dataset}.pt")