# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
import numpy as np
def extract_json_from_text(text):
    """
    Try to find and parse a JSON object from the given text.
    Handles cases where JSON is wrapped in ```json ... ```,
    or directly starts with { ... }, and also fixes single quotes.
    """
    if not isinstance(text, str):
        return None, False
    
    # Regex to find potential JSON blocks
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
    matches = json_pattern.findall(text)
    
    if not matches:  
        # fallback: look for { ... } directly
        brace_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if brace_match:
            matches = [brace_match.group(1)]
    
    if not matches:
        return None, False
    
    raw_json = matches[0].strip()
    
    # Try parsing with json.loads, fixing quotes if necessary
    try:
        return json.loads(raw_json), True
    except json.JSONDecodeError:
        try:
            # Replace single quotes with double quotes
            fixed_json = raw_json.replace("'", '"')
            return json.loads(fixed_json), True
        except json.JSONDecodeError:
            return None, True  # Found something but couldn’t parse

def extract_json_from_text2(text):
    """
    Extract the last JSON-like object from text.
    Handles cases where JSON is wrapped in ```json ... ```,
    or directly starts with { ... }, and also fixes single quotes.
    """
    if not isinstance(text, str):
        return None, False

    stack = []
    starts = []
    json_candidates = []

    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                starts.append(i)
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and starts:
                    start = starts.pop()
                    candidate = text[start:i+1]
                    json_candidates.append(candidate)

    if not json_candidates:
        return None, False

    # Take only the last JSON found
    raw_json = json_candidates[-1].strip()

    # Try parsing
    try:
        return json.loads(raw_json), True
    except json.JSONDecodeError:
        try:
            fixed_json = re.sub(r",(\s*[}\]])", r"\1", raw_json)
            return json.loads(fixed_json), True
        except json.JSONDecodeError:
            return None, True  # Found but invalid JSON

def process_dataframe(df, column="analysis_result"):
    results = []
    for text in df[column]:
        parsed_json, found_json = extract_json_from_text(text)
        
        if parsed_json:
            is_json = True
            phishing_score = parsed_json.get("phishing_score", np.nan)
            is_phishing = parsed_json.get("is_phishing", np.nan)
            reasoning = parsed_json.get("reasoning", np.nan)
        else:
            is_json = False
            phishing_score = np.nan
            is_phishing = np.nan
            reasoning = np.nan
        
        needs_processing = False
        if found_json and isinstance(text, str):
            # if JSON does not take the whole string, mark as needs_processing
            json_str = json.dumps(parsed_json) if parsed_json else ""
            needs_processing = len(text.strip()) != len(json_str.strip())
        
        results.append({
            "is_json": is_json,
            "needs_processing": needs_processing,
            "phishing_score": phishing_score,
            "is_phishing": is_phishing,
            "reasoning": reasoning
        })
    
    return df.join(pd.DataFrame(results))