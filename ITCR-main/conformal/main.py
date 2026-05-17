import numpy as np

import matplotlib.pyplot as plt
from math import ceil
import random
from tqdm import tqdm
import json
import networkx as nx

from conformal import get_prediction_set_new, calibrate_threshold, is_coherent_correct_subgraph
from train import load_split_features, load_graph_list

def compute_coverage_rate(test_graphs, lambda_hat, logreg, pos_token="Y"):
        N = len(test_graphs)

        cover_flags = []
        return_nodes_ratio = []

        for G in test_graphs:
            Gamma_i = get_prediction_set_new(G, lambda_hat, logreg)
            return_nodes = Gamma_i[-1]

            flag = True
            for idx,H_nodes in enumerate(Gamma_i):
                if not is_coherent_correct_subgraph(G, H_nodes, pos_token=pos_token, check_all_ancestors=False):
                    flag = False
                    break

            cover_flags.append(1 if flag else 0)
            return_nodes_ratio.append(len(return_nodes) / G.number_of_nodes())
        return sum(cover_flags) / N, np.mean(return_nodes_ratio)

if __name__ == "__main__":
    dataset = 'gsm8k' 
    interpolation = "lower" # "lower", "linear", "higher"

    import joblib
    focal_model = joblib.load(f"data/pre-trained/{dataset}/focal_model_{dataset}.pkl")

    calib_test_graphs = load_graph_list(f"data/pre-trained/{dataset}/calib_test_graphs_{dataset}.pkl")

    alpha_list = np.arange(0.1, 0, -0.05)
    coverage_rate_alpha = []
    return_nodes_ratio_alpha = []
    for alpha in tqdm(alpha_list):
        print("alpha: ", alpha)
        fact_list = []
        return_nodes_ratio_list = []
        lambda_hat_list = []
        n = 100
        for i in range(n):
            random.shuffle(calib_test_graphs, random.seed(0+i))
            calib_graphs = calib_test_graphs[:len(calib_test_graphs)//2]
            test_graphs = calib_test_graphs[len(calib_test_graphs)//2:]
            lambda_hat = calibrate_threshold(calib_graphs, alpha=alpha, logreg_pipeline=focal_model, pos_token="Y", interpolation=interpolation)
            lambda_hat_list.append(lambda_hat)
            coverage_rate, return_nodes_ratio = compute_coverage_rate(test_graphs, lambda_hat, focal_model, pos_token="Y")

            fact_list.append(coverage_rate)
            return_nodes_ratio_list.append(return_nodes_ratio)
        coverage_rate_mean = sum(fact_list) / n
        coverage_rate_std = np.std(fact_list)
        print(f"coverage_rate_mean_{dataset}_{alpha}: ", coverage_rate_mean)
        print(f"coverage_rate_std_{dataset}_{alpha}: ", coverage_rate_std)
        print(f"return_nodes_ratio_mean_{dataset}_{alpha}: ", np.mean(return_nodes_ratio_list))
        print(f"return_nodes_ratio_std_{dataset}_{alpha}: ", np.std(return_nodes_ratio_list))
        coverage_rate_alpha.append(coverage_rate_mean)
        return_nodes_ratio_alpha.append(np.mean(return_nodes_ratio_list))

    # print("mean lambda_hat: ", np.mean(lambda_hat_list))
    