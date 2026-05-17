import numpy as np
from scipy.stats import rankdata
import json
import re

CORRECT_ANNOTATIONS = ["Y"]

def query_model_system_llama(messages, pipeline, n_samples=1, temperature=0.1): 

    tokenizer = pipeline.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if n_samples == 1:
        outputs = pipeline(
            prompt,
            max_new_tokens=1000,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]
    else:

        prompts = [prompt] * n_samples
        
        all_outputs = pipeline(
            prompts,  
            max_new_tokens=1000,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            return_full_text=False,
            batch_size=n_samples,  
        )
        
        return [output[0]["generated_text"] for output in all_outputs]

def parse_jsonl_output(output):
    # Use regular expression to find all JSON objects in the output
    json_objects = re.findall(r"\{.*?\}", output, re.DOTALL)

    # Parse each JSON object and store in a list
    parsed_objects = []
    for obj in json_objects:
        try:
            # Ensure each JSON object is valid
            parsed_objects.append(json.loads(obj))
        except json.JSONDecodeError:
            continue

    return parsed_objects

def get_frequency_scores(pipeline, subclaims, prompt, n_samples):
    """
    Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
    """
    # Generate n_samples alternate outputs with temperature 1.0.
    messages = [{"role": "user", "content": prompt}]
    try:
        completion = query_model_system_llama(messages, pipeline, n_samples=n_samples, temperature=1.0)
    except Exception as e:
        print(f"Error generating alternate outputs: {e}")
        return [0.0] * len(subclaims)
    
    # Handle both single and multiple outputs
    if n_samples == 1:
        alternate_outputs = [completion]
    else:
        alternate_outputs = completion
    
    claim_string = "\n".join(
        [str(i) + ": " + fact["subclaim"] for i, fact in enumerate(subclaims)]
    )

    # Count the number of times the alternate outputs support the sub-claims (using LM).
    final_scores = [0.0] * len(subclaims)
    for output_text in alternate_outputs:
        counting_prompt = (
            'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with no explanation or other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
            + claim_string
            + "\n\nThe text is:\n"
            + output_text
        )
        counting_messages = [{"role": "user", "content": counting_prompt}]
        counting_output = query_model_system_llama(counting_messages, pipeline, n_samples=1, temperature=0.1)
        counting_output = counting_output.replace("```jsonl\n", "")
        counting_output = counting_output.replace("```", "")
        try:
            parsed_output = parse_jsonl_output(counting_output)
            for item in parsed_output:
                if "id" in item and "score" in item:
                    claim_id = int(item["id"])
                    score = int(item["score"])
                    if 0 <= claim_id < len(final_scores):
                        final_scores[claim_id] += score
        except Exception as ex:
            print(f"Failed to parse frequency score output: {ex}")
            continue

    return final_scores

def get_ranking(entry, confidence_method, use_percent=True):
    """
    Returns the corresponding ranking scores from the raw scores of confidence_method.
    """
    score_list = [
        -(subclaim[confidence_method + "-score"] + subclaim["noise"])
        for subclaim in entry["claims"]
    ]
    rankings = len(entry["claims"]) + 1 - rankdata(score_list, method="ordinal")
    if use_percent:
        rankings = rankings / len(entry["claims"])
    return rankings

def get_confidence(entry, method, pipeline):
    """
    Takes in an entry from {}_annotations.json and returns a list of confidence scores from method.
    """
    if method == "random":
        return [np.random.normal(0, 1) for subclaim in entry["claims"]]
    elif method == "baseline":
        return [
            len(entry["claims"]) - x for x in list(range(1, len(entry["claims"]) + 1))
        ]
    # elif method == "gpt":
    #     return [float(subclaim["gpt-score"]) for subclaim in entry["claims"]]
    elif method == "frequency":
        return get_frequency_scores(
            pipeline, entry["claims"], entry["prompt"], 5
        )
    # This assumes frequency was already added.
    # elif method == "frequency+gpt":
    #     return [
    #         subclaim["gpt-score"] + subclaim["frequency-score"]
    #         for subclaim in entry["claims"]
    #     ]
    elif method == "optimal":
        return [
            int(subclaim["annotation"] in CORRECT_ANNOTATIONS) # for gsm8k
            for subclaim in entry["claims"]
        ]
    # This assumes the corresponding raw scores were already added.
    # elif method in [
    #     "random-ranking",
    #     "baseline-ranking",
    #     # "gpt-ranking",
    #     "frequency-ranking",
    #     # "frequency+gpt-ranking",
    #     "optimal-ranking",
    # ]:
    #     return get_ranking(
    #         entry, method[:-8]
    #     )  # -8 is to remove '-ranking' from method.
    else:
        print(f"{method} method is not implemented.")