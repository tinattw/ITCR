import json
import re
from typing import Optional
import os
from tqdm import tqdm
import logging
from datetime import datetime

def extract_last_number(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.
    GSM8K-style: we take the last number occurring in the output.
    You can tighten this to parse 'Answer: ...' if you force that format.
    """
    m = re.search(r"Answer:\s*([^\n]+)", text)
    if m:
        ans = m.group(1)
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        if nums:
            return nums[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def extract_gsm8k_gold(answer_field: str) -> Optional[str]:
    """
    GSM8K gold often contains '#### <number>'.
    """
    m = re.search(r"####\s*([-]?\d+(?:\.\d+)?)", answer_field.replace(",", ""))
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_field.replace(",", ""))
    return nums[-1] if nums else None

def evaluate_jsonl(model, jsonl_path: str, max_n: int = None, save_results_path: str = None, log_path: str = None):
    total = 0
    correct = 0

    if log_path is None and save_results_path is not None:
        log_dir = os.path.dirname(save_results_path) if os.path.dirname(save_results_path) else "."
        log_filename = os.path.basename(save_results_path).replace(".json", ".log")
        log_path = os.path.join(log_dir, log_filename)
    
    if log_path:
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler() 
            ]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    
    if max_n is not None:
        total_lines = min(total_lines, max_n)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Evaluating", unit="sample"):
            if not line.strip():
                continue
            ex = json.loads(line)
            q = ex["question"]
            gold = extract_gsm8k_gold(ex["answer"])

            pred_text = model.generate(q)
            pred = extract_last_number(pred_text)
            

            is_correct = (pred is not None and gold is not None and pred == gold)

            total += 1
            correct += int(is_correct)

            if total > 0:
                status = 'Y' if is_correct else 'N'
                log_msg = f"Sample {total}/{total_lines}: {status} | Pred: {pred} | Gold: {gold}"
                tqdm.write(log_msg)
                
                if logger:
                    logger.info(log_msg)

            if max_n is not None and total >= max_n:
                break

    acc = correct / total if total else 0.0
    results = {
        "overall_score": acc,  
        "n": total,
        "correct": correct,
    }

    print("\n" + "="*50)
    print("Completed Evaluation")
    print("="*50)

    return results
