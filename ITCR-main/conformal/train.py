import os
import re
import numpy as np
from model import FocalLossClassifier
import torch
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score, log_loss
)
from data import load_graph_list
from conformal import calibrate_threshold
# from conformal import debug_one_graph
from conformal import feature_fn
from utils import pyg_to_features


def load_split_features(pt_path: str, k_eigs: int = 8, return_node_counts: bool = False):
    data_list = torch.load(pt_path, map_location="cpu")
    X = []
    y = []
    node_counts = []
    for d in data_list:
        X.append(pyg_to_features(d, k_eigs=k_eigs))
        # y expected shape [1]; binary 0/1 float
        yi = float(d.y.item()) if hasattr(d, "y") else 0.0
        y.append(int(yi >= 0.5))
        if return_node_counts:
            node_counts.append(d.x.shape[0] if hasattr(d, "x") and d.x is not None else 0)
    X = np.stack(X, axis=0) if X else np.zeros((0, k_eigs + 4), dtype=np.float32) # 4 is addtional node feature dim.
    y = np.array(y, dtype=np.int64)
    if return_node_counts:
        return X, y, np.array(node_counts, dtype=np.int64)
    return X, y


if __name__ == "__main__":
    dataset = 'gsm8k' 
    train_pt = f"data/pre-trained/{dataset}/train_pyg_{dataset}.pt"
    calib_test_pt = f"data/pre-trained/{dataset}/calib_test_pyg_{dataset}.pt"

    k_eigs = 8  # you can try 6/8/10; keep small to avoid overfitting

    X_train, y_train, node_counts_train = load_split_features(train_pt, k_eigs=k_eigs, return_node_counts=True)
    X_calib_test, y_calib_test, node_counts_calib = load_split_features(calib_test_pt, k_eigs=k_eigs, return_node_counts=True)

    print("Feature dim:", X_train.shape[1] if X_train.size else 0)
    print("Train:", X_train.shape, "pos_rate:", y_train.mean() if len(y_train) else None)
    print("Calib:", X_calib_test.shape, "pos_rate:", y_calib_test.mean() if len(y_calib_test) else None)

    input_dim = X_train.shape[1]
    
    pos_rate = y_train.mean()
    alpha = 1 - pos_rate  # less class weight higher

    if dataset == "gsm8k":
        dropout = 0
        hidden_dim=32 #
        epochs=500
    elif dataset == "MATH":
        dropout = 0
        hidden_dim=16       
        epochs=500
    elif dataset == "felm_wk":
        dropout = 0
        hidden_dim=32
        epochs=500

    focal_model = FocalLossClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        alpha=alpha,  # adjust by class imbalance degree
        gamma=2.0,    # focal loss focus parameter
        lr=0.001,
        batch_size=32,
        epochs=epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    focal_model.fit(X_train, y_train)

    import joblib
    joblib.dump(focal_model, f"data/pre-trained/{dataset}/focal_model_{dataset}.pkl")
