from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from parse import parse_step_json, convert_steps_to_json_format, parse_all_steps_json, parse_all_steps_json_rollout
from utils import load_graph_from_json, risk_scores_for_sequence
from utils import generate_growth_subgraphs
import json
import numpy as np

from get_scores import get_frequency_scores
from gen_graph import add_graphs, sequential_adjacency_list
import transformers

class ThresholdGatedLLM:

    def __init__(
        self,
        model,
        tokenizer,
        thr: float,
        max_steps: int = 10,
        max_tokens: int = 1000,
        do_sample: bool = True,
        temperature: float = 0.1,
        top_p: float = 0.9,
        device: str = "cuda",
        focal_model: object = None,
    ):
        self.model = model.eval()
        self.model = self.load_model()  

        self.tokenizer = tokenizer
        self.thr = thr
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.focal_model = focal_model

        self.last_interventions = 0
        self.last_extra_tokens = 0

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if hasattr(model, 'hf_device_map') and model.hf_device_map else None,
        )

    def load_model(self):
        return self.model

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    @torch.no_grad()
    def _render_generation_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prefix: str = "",
    ) -> str:
        system_prompt = system_prompt.strip()
        user_prompt = user_prompt.strip()

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return rendered + assistant_prefix
            except Exception as e:
                print(f"Warning: failed to apply chat template, falling back to raw prompt: {e}")

        prompt_parts = [part for part in (system_prompt, user_prompt) if part]
        raw_prompt = "\n\n".join(prompt_parts)
        return raw_prompt + assistant_prefix

    @torch.no_grad()  
    def _generate_text(self, prompt: str, max_tokens: int, response_prefix: str = "") -> Tuple[str, int]:
        """
        Generate text continuation and return (generated_text, new_tokens_count).
        """
        model = self.load_model()
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        
        gen_ids = model.generate(**inputs, **generate_kwargs)
        generated_ids = gen_ids[0][prompt_len:]
       
        cont = response_prefix + self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return cont

    @torch.no_grad()
    def _generate_text_rollout(self, prompt: str, max_tokens: int = 256, response_prefix: str = ""):
        model = self.load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=self.do_sample,
            temperature=(self.temperature if self.do_sample else None),
            top_p=(self.top_p if self.do_sample else None),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out_ids[0][prompt_len:]
        text = response_prefix + self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        if getattr(self, 'is_multi_gpu', False):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return text

    @torch.no_grad()
    def _calculate_frequency_scores_and_dep_graph(self, step_jsons_list, prompt):

        if not step_jsons_list:
            return step_jsons_list
        
        subclaims = []
        for step in step_jsons_list:
            subclaim_text = step.get("subclaim", "")
            if subclaim_text:  
                subclaims.append({"subclaim": subclaim_text})
        
        if subclaims:
            try:
                frequency_scores = get_frequency_scores(
                    pipeline=self.pipeline,
                    subclaims=subclaims,
                    prompt=prompt,  
                    n_samples=5
                )

                score_idx = 0
                for i, step in enumerate(step_jsons_list):
                    if step.get("subclaim"):  
                        if score_idx < len(frequency_scores):
                            step["gpt-score"] = frequency_scores[score_idx]
                            score_idx += 1
                        else:
                            step["gpt-score"] = 0

            except Exception as score_error:
                print(f"Warning: Failed to calculate frequency score: {score_error}")
                for step in step_jsons_list:
                    if "gpt-score" not in step:
                        step["gpt-score"] = 0

            question = {
                "prompt": prompt,  
                "claims": subclaims  
            }
            
            try:
                questions_with_graph = add_graphs([question])
                
                if questions_with_graph and len(questions_with_graph) > 0:
                    dep_graph_item = questions_with_graph[0].get("dep_graph", [])
                else:
                    dep_graph_item = sequential_adjacency_list(len(subclaims))

            except Exception as graph_error:
                print(f"Warning: Failed to generate dep_graph: {graph_error}")
                dep_graph_item = sequential_adjacency_list(len(subclaims))

            for i, step in enumerate(step_jsons_list):
                n_num = i + 1  
                
                subgraph = []
                for row_idx in range(n_num):
                    if row_idx < len(dep_graph_item):
                        subgraph.append(dep_graph_item[row_idx][:n_num])
                    else:
                        subgraph.append([0] * n_num)
                
                step["dep_graph"] = subgraph
        
        return step_jsons_list

    def generate(self, prompt: str) -> str:

        sys_prompt = (
            "Solve the problem step by step.\n"
            "Output exactly ONE line per step using the strict JSON format below:\n"
            'Step t: {"subclaim": "<one short reasoning step>", "gpt-score": <number>}\n'
            "Here, gpt-score is your confidence that this step is correctness. For the conf, return 1 for supports, -1 for contradicts, and 0 for unrelated\n"
            "Rules:\n"
            "- Use only the two keys: subclaim and gpt-score.\n"
            "- Do not output any extra text.\n"
            "- Do not output 'Answer:' until all steps are finished.\n"
            "Finally output exactly one line:\n"
            "Answer: <final numeric answer>\n\n"
        )

        initial_user_prompt = (
            f"Problem: {prompt.strip()}\n\n"
            # "Start immediately with Step 1: and do not say anything before it."
        )
        initial_prefix = 'Step 1: {"subclaim": "'
        base = self._render_generation_prompt(
            sys_prompt,
            initial_user_prompt,
            assistant_prefix=initial_prefix,
        )

        step_jsons_list = []  
        step_id = 0
        solution_lines = []

        # 1) generate a single step continuation
        cont =self._generate_text(
            base,
            max_tokens=self.max_tokens,
            response_prefix=initial_prefix,
        )

        try:
            all_steps = parse_all_steps_json(cont)
            if all_steps:
                step_jsons_list.extend(all_steps)
            else:
                step_json = parse_step_json(cont)
                if step_json:
                    step_jsons_list.append(step_json)

            if step_jsons_list:
                step_jsons_list = self._calculate_frequency_scores_and_dep_graph(step_jsons_list, prompt)

            # print("step_jsons_list:\n", step_jsons_list)

        except Exception as e:
            print(f"Warning: Failed to parse step JSON: {e}")
            step_jsons_list.append({"subclaim": cont[:200] if cont else "", "gpt-score": 0, "dep_graph": []})

        # Generate steps
        step_intervention_count = 0  

        i = 0
        while i < len(step_jsons_list):
            step_json = step_jsons_list[i]
            step_id = i + 1
            
            ori_data = convert_steps_to_json_format(step_jsons_list[:step_id], prompt)  
            dataset_score_attribute = "gpt-score"
            G = load_graph_from_json(ori_data, dataset_score_attribute)  
            seq = generate_growth_subgraphs(G)  
            smx, score = risk_scores_for_sequence(seq, self.focal_model)
            risk = score[-1]   
            print("risk:\n", risk)
            # ========================================================

            thr = self.thr

            max_interventions = 3

            if risk is not None and risk > thr and step_intervention_count < max_interventions: 
                rewrite_system_prompt = (
                    "Rewrite one math-solution step in strict JSON.\n"
                    'Output exactly one line in this format: Step t: {"subclaim": "<one short reasoning step>", "gpt-score": <number>}\n'
                    "Use only the keys subclaim and gpt-score.\n"
                    "gpt-score must be one of: 1, 0, -1.\n"
                    "Do not output explanations, markdown, or extra text."
                )

                rewrite_user_prompt = (
                    f"Problem: {prompt.strip()}\n\n"
                    + f"Step {step_id}: "
                    + json.dumps(step_json, ensure_ascii=False)
                    + "\n\n"
                    + "This step may be unreliable. Rewrite it more carefully.\n"
                    + f"Rewrite ONLY Step {step_id} to be correct and consistent with earlier steps.\n"
                    + f"Output exactly one line: Step {step_id}:"
                )
                rewrite_prefix = f'Step {step_id}: {{"subclaim": "'
                rewrite_prompt = self._render_generation_prompt(
                    rewrite_system_prompt,
                    rewrite_user_prompt,
                    assistant_prefix=rewrite_prefix,
                )
                rewrite_cont = self._generate_text(
                    rewrite_prompt, 
                    max_tokens=self.max_tokens, 
                    response_prefix=rewrite_prefix
                )
                try:
                    print("rewrite_cont:\n", rewrite_cont)
                    rewritten_json = parse_step_json(rewrite_cont)  
                    step_jsons_list[step_id-1] = rewritten_json 

                    prefix = step_jsons_list[0:step_id]     
                    next_step = step_id + 1

                    def make_rollout_prompt(problem_text: str,
                        prefix_steps: list,
                        next_step: int,
                        max_new_steps: int = 6) -> str:

                        prefix_lines = []
                        for i, s in enumerate(prefix_steps, start=1):
                            s_clean = {
                                "step": i,
                                "subclaim": s.get("subclaim", ""),
                                "gpt-score": int(s.get("gpt-score", 0)),
                                # "deps": s.get("deps", [])
                            }
                            prefix_lines.append(f'Step {i}: ' + json.dumps(s_clean, ensure_ascii=False))

                        prefix_jsonl = "\n".join(prefix_lines)

                        rollout_system_prompt = (
                            "Continue a math solution in strict JSONL.\n"
                            "Output only new steps and the final answer.\n"
                            'Each step line must be exactly: Step t: {"step": t, "subclaim": "...", "gpt-score": <number>}\n'
                            "Use only the keys step, subclaim, and gpt-score.\n"
                            "gpt-score must be one of: 1, 0, -1.\n"
                            "Do not modify or repeat the fixed prefix.\n"
                        )
                        rollout_user_prompt = (
                            f"{problem_text.strip()}\n\n"
                            "Fixed prefix steps:\n"
                            f"{prefix_jsonl}\n\n"
                            f"Continue from Step {next_step}.\n"
                            f"Step numbers must be consecutive integers starting at {next_step}.\n"
                            f"Generate at most {max_new_steps} additional steps.\n"
                            "After the final step, output exactly one line:\n"
                            "Answer: <final numeric answer>\n"
                            f"Start your first output line with exactly: Step {next_step}:"
                        )
                        rollout_prefix = (
                            f'Step {next_step}: {{"step": {next_step}, "subclaim": "'
                        )
                        rollout_prompt = self._render_generation_prompt(
                            rollout_system_prompt,
                            rollout_user_prompt,
                            assistant_prefix=rollout_prefix,
                        )
                        return rollout_prompt, rollout_prefix

                    problem = "Problem: " + prompt.strip() + "\n\n"
                    rollout_prompt, rollout_prefix = make_rollout_prompt(
                        problem, prefix, next_step
                    )

                    rollout_cont = self._generate_text_rollout(
                        rollout_prompt,
                        max_tokens=self.max_tokens,
                        response_prefix=rollout_prefix,
                    )
                    print("rollout_cont:\n", rollout_cont)
                    if "Answer" in rollout_cont:
                        break  

                    try:
                        step_json = parse_all_steps_json_rollout(rollout_cont, next_step)

                        step_jsons_list = prefix.copy()
                        if isinstance(step_json, list):
                            step_jsons_list.extend([s for s in step_json if isinstance(s, dict)])
                        elif isinstance(step_json, dict):
                            step_jsons_list.append(step_json)

                        step_jsons_list = self._calculate_frequency_scores_and_dep_graph(step_jsons_list, prompt)
                        print("step_new_jsons_list:\n", step_jsons_list)
                        i += 1
                    except Exception as e:
                        print(f"Warning: Failed to parse rollout JSON: {e}")
                        i += 1
                        continue

                except Exception as e:
                    print(f"Warning: Failed to parse rewritten step JSON: {e}")
                    i += 1
                    continue

            elif risk is not None and risk > thr and step_intervention_count >= max_interventions:
                print(f"Skipping intervention on step {step_id}: already intervened {step_intervention_count} times (max: {max_interventions})")
                i += 1
                continue
            else:
                i += 1
                continue

        final_step_lines = []
        for idx, step_json in enumerate(step_jsons_list, start=1):
            final_step_lines.append(
                f"Step {idx}: " + json.dumps(step_json, ensure_ascii=False)
            )
        answer_system_prompt = (
            "Return only the final numeric answer for the solved math problem.\n"
            "Output exactly one line in the form:\n"
            "Answer: <final numeric answer>\n"
        )
        answer_user_prompt = (
            f"Problem: {prompt.strip()}\n\n"
            "Completed solution steps:\n"
            + "\n".join(final_step_lines)
            + "\n\nReturn the final numeric answer only."
        )
        answer_prefix = "Answer: "
        answer_prompt = self._render_generation_prompt(
            answer_system_prompt,
            answer_user_prompt,
            assistant_prefix=answer_prefix,
        )
        ans_cont = self._generate_text(
            answer_prompt,
            max_tokens=self.max_tokens,
            response_prefix=answer_prefix,
        )
        ans_text = ans_cont.splitlines()[0].strip()
        if ans_text.startswith("Answer:"):
            ans_text = ans_text[len("Answer:") :].strip()
        solution_lines.append("Answer: " + ans_text)

        return "\n".join(solution_lines)

    

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]
