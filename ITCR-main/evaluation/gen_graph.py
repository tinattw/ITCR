import json
import os
import copy
import re
import pandas as pd
from tqdm import tqdm
from collections import deque
# from openai import OpenAI
import torch
import transformers

# GPT Graph Generation Prompt
def messages_to_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"[SYSTEM] {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"[USER] {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"[ASSISTANT] {msg['content']}\n"
    prompt += "[ASSISTANT] "
    return prompt

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

tokenizer = pipeline.tokenizer

def query_llama_chat(pipeline, system_prompt, user_prompt, max_new_tokens=1000):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,  
    )
    return out[0]["generated_text"]



graph_few_shot = """
    You are a system designed to create dependency graphs for subclaims in response to a given question. Your output must strictly adhere to the following instructions:

1. Graph Description:
   - Represent the dependency relationships between subclaims as a directed graph.
   - Each subclaim is a vertex in the graph.
   - An edge (b → a) exists if subclaim "a" depends on subclaim "b."
   - Subclaims that are "a priori" (e.g., assumptions or definitions) should not have any ancestors.

2. Output Format:
   - Provide your graph as an adjacency list of size NUM × NUM, where NUM is the number of subclaims (this will be given at the beginning of the prompt).
   -A template adjacency list with all entries zero will be included at the end of the prompt as reference. Your output should match this structure, but you should replace 0s with 1s where relevant, as described below.
   - Each entry in the adjacency list is a list of n integers:
     - A value of 1 at position i in row j indicates that subclaim j depends on subclaim i.
     - A value of 0 indicates no dependency.
     - Ensure no claim depends on itself (diagonal entries must be 0).

3. Rules:
   - The adjacency list must be square, with n rows and n columns, where n is the exact number of subclaims provided.
   - Each row and column must be exactly n integers. Do not include extra rows, columns, or misaligned entries.
   - The output must consist solely of the adjacency list (e.g., [[0,1,0],[0,0,1],[0,0,0]]); do not include explanations, commentary, or any other formatting.

4. Dependencies:
   - Consider explicit and implicit dependencies between subclaims. For example, if subclaim j implicitly relies on subclaim i (even if not stated directly), include the edge (i → j) in the graph.
   - Always represent dependencies, even if the subclaims are incorrect or contain logical errors.

Examples:

- Input:
  Question: How many vertical asymptotes does the graph of y = x / (x^2 + 1) have?

  NUM = 4
  Subclaims:
  1. A function has vertical asymptotes exactly where its denominator equals zero.
  2. To solve for the vertical asymptotes of the function y = x / (x^2 + 1), we therefore must solve x^2 + 1 = 0.
  3. For all real values of x, x^2 + 1 > 0.
  4. Thus, we conclude that the function y = x / (x^2 + 1) has no vertical asymptotes.

  Template:
  [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

  Desired Output:
  [[0,1,0,0], [0,0,1,1], [0,0,0,1], [0,0,0,0]]

- Input:
  Question: Consider the function y = x^2 + 2x + 15. What is the sum of the zeroes of this function?

  NUM = 5  
  Subclaims:
  1. The zeroes of a function are the x-values of its x-intercepts.
  2. To find the zeroes of y = x^2 + 2x + 15, we set the right-hand side equal to 0, writing 0 = x^2 + 2x + 15.
  3. To solve 0 = x^2 + 2x + 15, we factor it as 0 = (x+3)(x-5).
  4. This means that the zeroes of y = x^2 + 2x + 15 are x = -3, 5.
  5. We conclude that the sum of the zeroes of this function is -3 + 5 = 2.

  Template:
  [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

  Desired Output:
  [[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0]]

Now provide your adjacency list for the following question and subclaims:
"""


def generate_template(num):
    """
    Generates a template adjacency list of size num × num with all zeros.
    """
    return [[0] * num for _ in range(num)]

def sequential_adjacency_list(num_nodes):
    """
    Creates a sequential adjacency list with directed edges.
    Example for num_nodes = 3:
    [[0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]]
    """
    adj_list = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes - 1):
        adj_list[i][i + 1] = 1
    return adj_list

def query_model_system(
    client,
    system_prompt,
    user_prompt,
    model,
    max_tokens=2000,
    temperature=0,
    n_samples=1,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n_samples,
    )
    return completion.choices[0].message.content

def parse_graph_output(output):
    """
    Parses the output from the model to extract the adjacency list.
    Ensures the adjacency list is a list of lists containing only 0 or 1.
    Replaces non-zero entries with 1 if invalid values are found.
    """
    # Remove any explanation or commentary before the list of lists
    output = re.sub(r"^[^\[]*\[", "[", output, flags=re.DOTALL)

    # Remove any newlines or extra spaces within the list of lists
    output = re.sub(r"\s+", "", output)

    try:
        # Evaluate the cleaned output to get the adjacency list
        adjacency_list = eval(output)

        # Ensure the adjacency list is a list of lists
        if all(isinstance(row, list) for row in adjacency_list):
            # Replace all non-zero entries with 1
            adjacency_list = [
                [0 if i == 0 else 1 for i in row] for row in adjacency_list
            ]
            return adjacency_list
    except (SyntaxError, ValueError):
        pass

    # Return None if validation fails
    return None

def is_cyclic(adj_list):
    """
    Detects if a graph is cyclic using Kahn's algorithm.
    Returns True if graph is cyclic, False o.w.
    """
    in_degree = {i: 0 for i in range(len(adj_list))}

    # Compute in-degrees of all nodes
    for i in range(len(adj_list)):
        for j in range(len(adj_list[i])):
            if adj_list[i][j] == 1:
                in_degree[j] += 1

    # Initialize a queue and enqueue all nodes with in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1

        # Reduce in-degree of adjacent nodes by 1
        for i in range(len(adj_list[node])):
            if adj_list[node][i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    # If all nodes are not visited, there's a cycle
    return visited_count != len(adj_list)

# def add_graphs(questions, client, model):
def add_graphs(questions):
    """
    Queries model to generate deducibility graph proxies for responses.
    If the generated graph is cyclic, replaces it with a sequential adjacency list.
    """
    updated_questions = copy.deepcopy(questions)

    for question in tqdm(updated_questions, desc="Processing Questions"):
        # Define the system prompt
        system_prompt = graph_few_shot

        # Define the user prompt
        num_subclaims = len(question["claims"])
        user_prompt = (
            "Question: " + question["prompt"] + f"\n\nNUM: {num_subclaims}\nSubclaims:"
        )

        # Add subclaims to the prompt
        for j, claim in enumerate(question["claims"], start=1):
            user_prompt += f"\n{j}. " + claim["subclaim"]

        # Generate the template adjacency list
        template = generate_template(num_subclaims)
        user_prompt += f"\n\nTemplate:\n{template}"

        max_attempts = 5
        attempts = 0
        valid_graph = False
        parsed_graph = None

        while attempts < max_attempts and not valid_graph:
            
            dep_graph = query_llama_chat(pipeline, system_prompt, user_prompt)

            # Parse and validate the graph
            parsed_graph = parse_graph_output(dep_graph)

            if parsed_graph and len(parsed_graph) == num_subclaims:
                valid_graph = True
            else:
                attempts += 1

        if valid_graph:
            try:
                if is_cyclic(parsed_graph):
                    # Replace cyclic graph with sequential adjacency list
                    question["dep_graph"] = sequential_adjacency_list(num_subclaims)
                else:
                    # Store the valid acyclic graph
                    question["dep_graph"] = parsed_graph
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error checking graph for cycles: {e}. Using sequential graph.")
                question["dep_graph"] = sequential_adjacency_list(num_subclaims)
        else:
            print(
                f"Failed to generate valid graph after {max_attempts} attempts for question: {question['prompt']}"
            )
            print(user_prompt)
            # Replace with sequential graph if unable to generate a valid graph
            question["dep_graph"] = sequential_adjacency_list(num_subclaims)

    return updated_questions