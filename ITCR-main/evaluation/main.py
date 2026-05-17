from concurrent.futures import thread
from pathlib import Path
import sys

from evaluation import evaluate_jsonl
from llm_model import ThresholdGatedLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import joblib

# Pickled focal_model expects `import model` (see conformal/train.py); mirror training sys.path.
_conformal_dir = Path(__file__).resolve().parent.parent / "conformal"
sys.path.insert(0, str(_conformal_dir))
import numpy as np
import json
import io
from datasets import load_dataset
import datetime

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    dataset = "gsm8k"

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,     
        device_map="auto",             
        low_cpu_mem_usage=True,
    )
    model.eval()

    focal_model = joblib.load(f"data/pre-trained/{dataset}/focal_model_{dataset}.pkl") 

    thr_05 = 0.28260702 # pre-trained threshold when alpha=0.05 for gsm8k
    thr_1 = 0.5012 # pre-trained threshold when alpha=0.1 for gsm8k
    thr_2 = 0.6765 # pre-trained threshold when alpha=0.2 for gsm8k

    gated_model = ThresholdGatedLLM(model=model, tokenizer=tokenizer, thr=thr_05, do_sample=True, focal_model=focal_model)

    res = evaluate_jsonl(
        model=gated_model,
        jsonl_path=f"data/{dataset}_open_ori_500.jsonl",
        save_results_path=f"results/{dataset}_thr_05_{now}.json"
    )
    print(res)