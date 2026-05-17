from pickle import TRUE
from typing import List, Tuple, Any
import numpy as np
import networkx as nx
from math import ceil

from data import nx_to_pyg_data_fixed
from utils import pyg_to_features

import torch
import warnings
import math

def feature_fn(subG: nx.DiGraph, k_eigs: int = 8) -> np.ndarray:
    pyg = nx_to_pyg_data_fixed(subG)           
    feats = pyg_to_features(pyg, k_eigs=k_eigs)
    feats = np.asarray(feats, dtype=np.float32)
    return feats

def calculate_conformal_value(scores, alpha, interpolation, default_q_hat = torch.inf):
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    qunatile_value = math.floor((N + 1) * (alpha)) / N
    if qunatile_value > 1:
        warnings.warn(
            f"The value of quantile exceeds 1. It should be a value in [0,1]. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat

    return torch.quantile(scores, qunatile_value, dim=0, interpolation=interpolation).to(scores.device)

def calculate_conformal_value_no_missing(dataset, scores, alpha, interpolation, default_q_hat = torch.inf):
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    qunatile_value = math.ceil((N + 1) * (1 - alpha)) / N
    if dataset == "MATH":
        qunatile_value = math.ceil((N) * (1 - alpha)) / N
    if qunatile_value > 1:
        warnings.warn(
            f"The value of quantile exceeds 1. It should be a value in [0,1]. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat

    return torch.quantile(scores, qunatile_value, dim=0, interpolation=interpolation).to(scores.device)


def generate_growth_subgraphs(
    G: nx.DiGraph,
    use_prefix: bool = True,
    max_steps: int = None,
) -> List[nx.DiGraph]:
    """
    Generate a sequence of subgraphs by topological prefix growth.
    """
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


def is_coherent_correct_subgraph(G: nx.DiGraph, sub_nodes: List[Any], pos_token: str = "Y", check_all_ancestors: bool = True) -> bool:
    sub = set(sub_nodes)
    if len(sub) == 0:
        return True

    for v in sub:
        if G.nodes[v].get("annotation", None) != pos_token:
            return False

    if check_all_ancestors:
        for v in sub:
            if not nx.ancestors(G, v).issubset(sub):
                return False
    else:
        for v in sub:
            for p in G.predecessors(v):
                if p not in sub:
                    return False

    return True

def graph_size_penalty(n_t: int, max_nodes_cap: int = 64, mode="linear") -> float:
    if mode == "log":
        return np.log1p(n_t) / np.log1p(max_nodes_cap)
    return n_t / max_nodes_cap

def risk_scores_for_sequence(
    subgraphs: List[nx.DiGraph],
    logreg_pipeline,
    feature_fn,
    beta: float = 0.25,
    max_nodes_cap: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if not subgraphs:
        return np.array([]), np.array([])

    X = np.stack([feature_fn(sg) for sg in subgraphs], axis=0)  # (T,d)

    smx = logreg_pipeline.predict_proba(X)[:, 0]
    size_terms = np.array([len(sg.nodes())-1 for sg in subgraphs])

    score = smx + beta * size_terms

    score[0] = -np.inf

    return smx, score

def risk_scores_for_sequence_no_missing(
    subgraphs: List[nx.DiGraph],
    logreg_pipeline,
    feature_fn,
    beta: float = 0.25,
    max_nodes_cap: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:

    if not subgraphs:
        return np.array([]), np.array([])

    X = np.stack([feature_fn(sg) for sg in subgraphs], axis=0)  # (T,d)

    smx = logreg_pipeline.predict_proba(X)[:, 1]
    size_terms = np.array([len(sg.nodes())-1 for sg in subgraphs])

    score = 1 - smx + beta * size_terms

    score[0] = -np.inf

    return smx, score

def calibrate_threshold(
    calibration_graphs: List[nx.DiGraph],
    alpha: float,
    logreg_pipeline,
    pos_token: str = "Y",
    interpolation: str = "linear",
) -> float:
    score_list = []

    for G in calibration_graphs:
        seq = generate_growth_subgraphs(G)

        smx, score = risk_scores_for_sequence(seq, logreg_pipeline, feature_fn)
        # print("full score list gap: ", [score[i] - score[i+1] for i in range(len(score)-1)]) 

        for t, subG in enumerate(seq):
            if not is_coherent_correct_subgraph(G, list(subG.nodes()), pos_token=pos_token, check_all_ancestors=True):
                score_list.append(float(score[t]))
        # print("correct score list: ", score[:last_ok+1])
        # print("correct score list gap: ", [score[i+1] - score[i] for i in range(last_ok)])

    if len(score_list) == 0:
        raise ValueError("Calibration score_list is empty; check labels or growth strategy.")

    score_list = torch.tensor(score_list)
    # print("score_list: ", score_list)

    lambda_hat = calculate_conformal_value(score_list, alpha, interpolation=interpolation, default_q_hat = torch.inf)
    return lambda_hat

def calibrate_threshold_no_missing(
    dataset: str,
    calibration_graphs: List[nx.DiGraph],
    alpha: float,
    logreg_pipeline,
    pos_token: str = "Y",
    interpolation: str = "linear",
) -> float:
    score_list = []

    for G in calibration_graphs:
        seq = generate_growth_subgraphs(G)


        smx, score = risk_scores_for_sequence_no_missing(seq, logreg_pipeline, feature_fn)
        correct_list = []
        for t, subG in enumerate(seq):
            if is_coherent_correct_subgraph(G, list(subG.nodes()), pos_token=pos_token, check_all_ancestors=True):
                correct_list.append(t)
        score_list.append(float(score[correct_list[-1]]))

    if len(score_list) == 0:
        raise ValueError("Calibration score_list is empty; check labels or growth strategy.")

    score_list = torch.tensor(score_list)

    lambda_hat = calculate_conformal_value_no_missing(dataset, score_list, alpha, interpolation=interpolation, default_q_hat = torch.inf)
    return lambda_hat

def get_prediction_set_new(
    G: nx.DiGraph,
    threshold: float,
    logreg_pipeline=None,
    k_eigs: int = 8,
    max_steps: int = None,
    return_graphs: bool = False,
    return_scores: bool = False,
):

    seq = generate_growth_subgraphs(G, max_steps=max_steps)
    
    smx, score = risk_scores_for_sequence(seq, logreg_pipeline, feature_fn)

    t_stop = None
    for t in range(len(seq)):
        if score[t] > threshold:
            t_stop = t
            break

    if t_stop is None:
        t_star = len(seq) - 1
    else:
        t_star = max(t_stop - 1, 0)

    if return_graphs:
        return seq[: t_star + 1]

    Gamma_nodes = [list(seq[t].nodes()) for t in range(t_star + 1)]

    if return_scores:
        return Gamma_nodes, score[: t_star + 1]
        
    return Gamma_nodes

def get_prediction_set_no_missing(
    G: nx.DiGraph,
    threshold: float,
    logreg_pipeline=None,
    k_eigs: int = 8,
    max_steps: int = None,
    return_graphs: bool = False,
    return_scores: bool = False,
):

    seq = generate_growth_subgraphs(G, max_steps=max_steps)

    smx, score = risk_scores_for_sequence_no_missing(seq, logreg_pipeline, feature_fn)

    t_stop = None
    for t in range(len(seq)):
        if score[t] > threshold:
            t_stop = t
            break

    if t_stop is None:
        t_star = len(seq) - 1
    else:
        t_star = max(t_stop - 1, 0)

    if return_graphs:
        return seq[: t_star + 1]

    Gamma_nodes = [list(seq[t].nodes()) for t in range(t_star + 1)]

    seq_nodes = [list(seq[t].nodes()) for t in range(len(seq) - 1 + 1)]

    if return_scores:
        return seq_nodes, Gamma_nodes, score[: t_star + 1]
    return seq_nodes,Gamma_nodes

