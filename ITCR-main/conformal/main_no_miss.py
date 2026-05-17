import numpy as np

import matplotlib.pyplot as plt
from math import ceil
import random
from tqdm import tqdm
import json
import networkx as nx

from conformal import get_prediction_set_no_missing, calibrate_threshold_no_missing, is_coherent_correct_subgraph
from train import load_split_features, load_graph_list

def compute_coverage_rate_no_missing(test_graphs, lambda_hat, logreg, pos_token="Y"):
        N = len(test_graphs)

        cover_flags = []
        return_nodes_ratio = []
        for G in test_graphs:
            seq_nodes, Gamma_i, scores = get_prediction_set_no_missing(G, lambda_hat, logreg, return_scores=True)
            return_nodes = Gamma_i[-1]

            # print("test scores: ", scores)

            t = len(Gamma_i) - 1 # index of the last generated subgraph
            correct_list = []
            for idx,H_nodes in enumerate(seq_nodes):
                # print(f"H_nodes: {H_nodes}")
                if is_coherent_correct_subgraph(G, H_nodes, pos_token=pos_token, check_all_ancestors=False): # traverse all subgraphs, check if all coherent correct subgraphs are included
                    correct_list.append(idx)
            cover_flags.append(1 if correct_list[-1] <= t else 0)
            return_nodes_ratio.append(len(return_nodes) / G.number_of_nodes())
        return sum(cover_flags) / N, np.mean(return_nodes_ratio)


if __name__ == "__main__":
    dataset = 'gsm8k' 
    interpolation = "linear" # "lower", "linear", "higher"
    import joblib
    focal_model = joblib.load(f"data/pre-trained/{dataset}/focal_model_{dataset}.pkl")

    calib_test_graphs = load_graph_list(f"data/pre-trained/{dataset}/calib_test_graphs_{dataset}.pkl")

    alpha_list = np.arange(0.1, 0, -0.05)
    coverage_rate_alpha = []
    return_efficiency_alpha = []
    for alpha in tqdm(alpha_list):
        print("alpha: ", alpha)

        fact_list = []
        return_nodes_ratio_list = []
        n = 100
        for i in range(int(n)):
            random.shuffle(calib_test_graphs, random.seed(0+i))
            calib_graphs = calib_test_graphs[:len(calib_test_graphs)//2]
            test_graphs = calib_test_graphs[len(calib_test_graphs)//2:]
            lambda_hat = calibrate_threshold_no_missing(dataset, calib_graphs, alpha=alpha, logreg_pipeline=focal_model, pos_token="Y", interpolation=interpolation)
            coverage_rate, return_nodes_ratio = compute_coverage_rate_no_missing(test_graphs, lambda_hat, focal_model, pos_token="Y")

            fact_list.append(coverage_rate)
            return_nodes_ratio_list.append(return_nodes_ratio)
        coverage_rate_mean = sum(fact_list) / n
        return_efficiency = 1-sum(return_nodes_ratio_list) / n
        coverage_rate_std = np.std(fact_list)
        return_nodes_ratio_std = np.std(return_nodes_ratio_list)
        coverage_rate_alpha.append(coverage_rate_mean)
        return_efficiency_alpha.append(return_efficiency)
        print(f"coverage_rate_mean_{dataset}_{alpha}: ", coverage_rate_mean)
        print(f"coverage_rate_std_{dataset}_{alpha}: ", coverage_rate_std)
        print(f"return_efficiency_{dataset}_{alpha}: ", return_efficiency)
        print(f"return_nodes_ratio_std_{dataset}_{alpha}: ", return_nodes_ratio_std)
    