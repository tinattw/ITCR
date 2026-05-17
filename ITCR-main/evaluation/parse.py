import json, re
from typing import List, Dict, Optional
import numpy as np

def parse_step_json(text: str, default_value: dict = None):
    if default_value is None:
        default_value = {"subclaim": "", "gpt-score": 0, "deps": []}

    text_clean = text.lstrip()
    if not text_clean:
        return default_value
    
    start = text_clean.find("{")
    if start == -1:
        return default_value

    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text_clean)):
        char = text_clean[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == "\\":
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_str = text_clean[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        try:
                            fixed_json = re.sub(r'(\w+):', r'"\1":', json_str)
                            return json.loads(fixed_json)
                        except:
                            return default_value

    return default_value

def convert_steps_to_json_format(
    step_jsons: List[Dict], 
    prompt: str, 
    original_output: Optional[str] = None
) -> Dict:
    
    num_steps = len(step_jsons)
    
    claims = []
    for step_json in step_jsons:
        claim = {
            "subclaim": step_json.get("subclaim", ""),
            "noise": np.random.normal(0, 0.001),  
            "gpt-score": step_json.get("gpt-score", 0),  
        }
        claims.append(claim)
    
    dep_graph = [[0] * num_steps for _ in range(num_steps)]
    
    for i, step_json in enumerate(step_jsons):
        deps = step_json.get("deps", [])
        for dep_idx in deps:
            if isinstance(dep_idx, int) and 1 <= dep_idx <= num_steps:
                dep_graph[i][dep_idx - 1] = 1
    
    result = {
        "prompt": prompt,
        "claims": claims,
        "dep_graph": dep_graph
    }
    
    if original_output:
        result["original-output"] = original_output
    
    return result

def parse_all_steps_json(text: str) -> List[Dict]:

    results = []
    text_clean = text.strip()
    
    if not text_clean:
        return results
    
    lines = text_clean.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        step_json = parse_step_json(line)
        if step_json and step_json.get("subclaim"):
            results.append(step_json)
    
    if not results:
        remaining_text = text_clean
        while True:
            step_json = parse_step_json(remaining_text)
            if not step_json or not step_json.get("subclaim"):
                break
            
            results.append(step_json)
            
            start = remaining_text.find("{")
            if start == -1:
                break
            
            depth = 0
            in_string = False
            escape_next = False
            end_pos = -1
            
            for i in range(start, len(remaining_text)):
                char = remaining_text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == "\\":
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            end_pos = i + 1
                            break
            
            if end_pos == -1:
                break
            
            remaining_text = remaining_text[end_pos:].lstrip()
            if not remaining_text:
                break
    
    return results

def parse_all_steps_json_rollout(text: str, next_step: int) -> List[Dict]:

    results = []
    text_clean = text.strip()
    
    if not text_clean:
        return results
    
    lines = text_clean.split('\n')
    
    start_idx = -1
    pattern = re.compile(rf'^Step\s+{next_step}\s*:', re.IGNORECASE)
    
    for i, line in enumerate(lines):
        if pattern.match(line.strip()):
            start_idx = i
            break
    
    if start_idx == -1:
        for i, line in enumerate(lines):
            if re.search(r'Step\s+\d+\s*:', line, re.IGNORECASE):
                start_idx = i
                break
    
    if start_idx == -1:
        for i, line in enumerate(lines):
            if '{' in line:
                start_idx = i
                break
    
    if start_idx == -1:
        return results
    
    remaining_text = '\n'.join(lines[start_idx:])
    
    step_pattern = re.compile(r'Step\s+(\d+)\s*:\s*(\{.*?\})', re.DOTALL | re.IGNORECASE)
    matches = list(step_pattern.finditer(remaining_text))
    
    for match in matches:
        step_num = int(match.group(1))
        json_str = match.group(2)
        
        if step_num >= next_step:
            try:
                step_json = json.loads(json_str)
                if step_json.get("subclaim"):  
                    results.append(step_json)
            except json.JSONDecodeError:
                step_json = parse_step_json(json_str)
                if step_json and step_json.get("subclaim"):
                    results.append(step_json)
    
    if not results:
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            if re.search(r'Step\s+\d+\s*:', line, re.IGNORECASE):
                step_json = parse_step_json(line)
                if step_json and step_json.get("subclaim"):
                    results.append(step_json)
    
    return results