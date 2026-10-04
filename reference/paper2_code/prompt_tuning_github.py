# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random

import pandas as pd
from anthropic import AnthropicFoundry

# Add project scripts to Python path
ROOT = ""
sys.path.append(ROOT)

from extract_json_github import *
from local_llm_infer_github import *
from prompt_teacher_tuning_github import *
from prompt_initial_text_only_github import *
from prompt_initial_screenshot_only_github import *
from prompt_initial_text_screenshot_github import *

# ---- Models  ----
endpoint = ""
deployment_name = ""
api_key = ""

TEACHER_MODEL_client = AnthropicFoundry(
    api_key=api_key,
    base_url=endpoint
)

CONDITION = "screenshot"

# ---- Paths ----
DATA_FOLDER = ""      # <id>.json summaries
TRAIN_CSV   = ""        # id,label,filename  (shuffled)
OUT_DIR     = ""
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Tuning hyperparameters ----
DEV_SIZE          = 1000     # fixed dev set carved from the front of train.csv
BATCH_SIZE        = 1000     # student training batch size per round
MAX_ERRORS_SHOWN  = 60       # cap misclassified cases sent to the teacher
MAX_CORRECT_SHOWN = 20       # anchor of correct near-boundary cases
IMPROVE_MARGIN    = 0.02     # candidate must beat best dev by this to be adopted
SEED              = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(site_id):
    """Load one site's JSON summary as a structure-preserving string."""
    with open(os.path.join(DATA_FOLDER, f"{site_id}.json"),
              "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2, ensure_ascii=False)

not_phish_all = pd.read_csv("")
phish_all = pd.read_csv("")

model_list = ["nemotron3:33b"]

# ---------------------------------------------------------------------------
# Running the student over a set of sites
# ---------------------------------------------------------------------------

def run_student(model_list, mode, prompt, rows):
    """Run the student on the given rows (DataFrame rows: id, label).
    
    mode: ["Text", "Screenshot", "Both"]
    
    prompt_builder_or_text is EITHER:
      * a callable taking html_summary -> full prompt string (the initial round), OR
      * a plain prompt string with a {html_summary} placeholder (teacher-produced
        prompts), filled per site.
 
    CONDITION (module-level switch) selects what the student sees:
      "text"  -> summarised html + url, text-only inference  (local_llm_infer_v2)
      "image" -> screenshot only,       image inference       (local_llm_infer_v3)
      "both"  -> summarised html + url + screenshot           (local_llm_infer_v3)
 
    Returns (records, metrics). records carry per-site verdict + reasoning.
    """
    if callable(prompt):
        prompt_template = prompt("{html_summary}")
    else:
        prompt_template = prompt
    
    for m in model_list:
        result_collection = {}  # stores results per model
    
        if not m in result_collection.keys():
            result_collection[m] = {}
        
        print(m)
        print(time.time())
        
        pageID = 0  # unique ID per webpage

        # Iterate over dataset entries in the order given by the CSV
        for _, row in rows.iterrows():
            ws_name = row["_id"]
            true_label = bool(row["label"])  # label comes from the CSV
                
            # Initialize results for this page
            if not pageID in result_collection[m].keys():
                result_collection[m][pageID] = {}
                
            # Load the site's prompt input (JSON summary or raw HTML)
            html_summary = load_summary(ws_name)
            
            # Load Prompt
            html_prompt = prompt_template.replace("{html_summary}", html_summary)
            
            run_result = {}
            
            if mode in ["screenshot", "both"]:
                if true_label == True:
                    image_folder = "/workspace/unpacked_folder_phishing"
                else:
                    image_folder = "/workspace/unpacked_folder_not_phishing"
                
                target_folder = os.path.join(image_folder, ws_name)
                
                # Open screenshot image (assumed .jpg in folder)
                png_file = [f for f in os.listdir(target_folder) if f.lower().endswith(".jpg")][0]
                image_path = os.path.join(target_folder, png_file)
                
                # Run model inference and time it
                t1 = time.time()
                analysis_result = local_llm_infer_v3(html_prompt, image_path, max_tokens=1500, model=m)
                t2 = time.time()
                
            else:
                # Run model inference and time it
                t1 = time.time()
                analysis_result = local_llm_infer_v2(html_prompt, max_tokens=1500, model=m)
                t2 = time.time()
                
            # Store results
            run_result["ws_name"] = ws_name
            run_result["True_Phish_Label"] = true_label
            run_result["runtime"] = t2 - t1
            run_result["prompt_len"] = len(html_prompt)
            run_result["analysis_result"] = analysis_result
            
            result_collection[m][pageID] = run_result    
                
            pageID += 1        
                
    return result_collection

def unpack_results(d, model):
    res = d[model]
    
    df = pd.DataFrame(res).T.reset_index(drop=True)

    parsed = df["analysis_result"].apply(extract_json_from_text2)
    json_df = pd.json_normalize(
        parsed.apply(lambda x: x[0] if x[1] and x[0] is not None else {})
    ).reset_index(drop=True)
    
    res_df = pd.concat([df, json_df], axis=1)
    res_df["True_Phish_Label"] = res_df["True_Phish_Label"].astype(bool)
    res_df["is_phishing"] = res_df["is_phishing"].astype(bool)
    
    return res_df

def compute_metrics(records):
    truth = records["True_Phish_Label"]
    pred  = records["is_phishing"]

    tp = int(( truth &  pred).sum())
    fn = int(( truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth &  pred).sum())

    n_phish, n_benign, n = tp + fn, tn + fp, len(records)
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "tpr": tp / n_phish if n_phish else 0.0,
        "fnr": fn / n_phish if n_phish else 0.0,
        "tnr": tn / n_benign if n_benign else 0.0,
        "fpr": fp / n_benign if n_benign else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_total": n, "n_phish": n_phish, "n_benign": n_benign,
    }


# ---------------------------------------------------------------------------
# Building the teacher's evidence package (error-focused + correct anchor)
# ---------------------------------------------------------------------------

def build_reasons(records, seed=SEED):
    """Select misclassified cases + a small anchor of correct near-boundary
    cases from the unpacked results dataframe. Keeps the teacher's context
    small and bias-controlled.
 
    Expects columns: True_Phish_Label (bool), is_phishing (bool),
    phishing_score, reasoning.
    """
    correct_mask = records["True_Phish_Label"] == records["is_phishing"]
    errors = records[~correct_mask]
    correct = records[correct_mask]
 
    # Near-boundary correct cases = middling confidence (not the easy 0/9s).
    near = correct[correct["phishing_score"].between(3, 6)]
    correct_anchor = near if len(near) else correct
 
    errors = errors.sample(min(len(errors), MAX_ERRORS_SHOWN), random_state=seed)
    correct_anchor = correct_anchor.sample(
        min(len(correct_anchor), MAX_CORRECT_SHOWN), random_state=seed)
 
    cols = ["True_Phish_Label", "is_phishing", "phishing_score", "reasoning"]
    return {
        "errors": errors[cols].to_dict("records"),
        "correct": correct_anchor[cols].to_dict("records"),
        "error_patterns": [],   # optional: fill with clustered reasoning phrases
    }



# ---------------------------------------------------------------------------
# The tuning loop: hill-climb with rejection on a FIXED dev set
# ---------------------------------------------------------------------------

# ---- Run the tuning loop (flat script, runs top-to-bottom in Spyder) ----
df = pd.read_csv(TRAIN_CSV)

# Fixed dev set (carved once, never trained on) + training pool.
dev_df = df.iloc[:DEV_SIZE].reset_index(drop=True)
train_pool = df.iloc[DEV_SIZE:].reset_index(drop=True)

# Training batches of BATCH_SIZE (last batch takes the remainder).
batches = [train_pool.iloc[i:i + BATCH_SIZE]
           for i in range(0, len(train_pool), BATCH_SIZE)]

print(f"dev: {len(dev_df)} | train pool: {len(train_pool)} | batches: {len(batches)}")

# --- Initial prompt: convert the initial builder into a reusable template ---
# We run round 0 with the callable builder; thereafter the prompt is a string
# with a {html_summary} placeholder that the teacher rewrites.

if CONDITION == "text":
    best_prompt = build_html_prompt_v6_text("{html_summary}")  # template form
elif CONDITION == "screenshot":
    best_prompt = build_html_prompt_v6_t_sc("{html_summary}")
elif CONDITION == "both":
    best_prompt = build_html_prompt_v6_sc()
# Score the initial prompt on the fixed dev set.
best_dev = run_student(["nemotron3:33b"], CONDITION, best_prompt, dev_df)

save_loc = os.path.join(OUT_DIR, f"nemotron3_33b_init_{CONDITION}.json")
with open(save_loc, 'w') as fp:
    json.dump(best_dev, fp) 
    
best_dev_df = unpack_results(best_dev, "nemotron3:33b")

best_metr = compute_metrics(best_dev_df)
best_dev_score = best_metr["accuracy"]
print(f"[init] dev accuracy = {best_dev_score:.4f}")
 
history = [{
    "round": 0, "adopted": True, "dev_accuracy": best_dev_score,
    "dev_metrics": best_metr, "prompt": best_prompt,
}]
 
for k, batch in enumerate(batches, start=1):
    print(f"\n=== Round {k}/{len(batches)} (batch of {len(batch)}) ===")
 
    # 1) Run the CURRENT BEST prompt on this training batch, collect evidence.
    train_raw = run_student(["nemotron3:33b"], CONDITION, best_prompt, batch)
    train_df = unpack_results(train_raw, "nemotron3:33b")
    train_metrics = compute_metrics(train_df)
    reasons = build_reasons(train_df)
 
    # 2) Teacher proposes a candidate prompt from the best prompt + evidence.
    teacher_prompt = teacher_prompt(best_prompt, train_metrics, reasons, modality=CONDITION)    
    response = TEACHER_MODEL_client.messages.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": teacher_prompt
            }
        ],
        max_tokens=8192,
    )
    teacher_response = response.content[0].text
    raw = teacher_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        candidate_prompt = json.loads(raw).get("new_prompt", "")
    except json.JSONDecodeError:
        candidate_prompt = ""
 
    if not candidate_prompt.strip():
        print("  teacher returned no new_prompt; keeping best, skipping.")
        history.append({"round": k, "adopted": False,
                        "reason": "empty candidate"})
        continue
    else:
        print("Continuing with new prompt")
 
    # 3) Score the candidate on the FIXED dev set.
    cand_raw = run_student(["nemotron3:33b"], CONDITION, candidate_prompt, dev_df)
    save_loc = os.path.join(OUT_DIR, f"nemotron3_33b_batch_{k}_{CONDITION}.json")
    with open(save_loc, 'w') as fp:
        json.dump(best_dev, fp)
    cand_dev_df = unpack_results(cand_raw, "nemotron3:33b")
    cand_dev = compute_metrics(cand_dev_df)
    cand_score = cand_dev["accuracy"]
    improved = cand_score >= best_dev_score + IMPROVE_MARGIN
 
    print(f"  train acc (best prompt): {train_metrics['accuracy']:.4f}")
    print(f"  dev acc (candidate):     {cand_score:.4f}  "
          f"(best so far {best_dev_score:.4f}) -> "
          f"{'ADOPT' if improved else 'reject'}")
 
    # 4) Adopt only on a real margin; always advance to the next batch.
    if improved:
        best_prompt = candidate_prompt
        best_dev_score = cand_score
 
    history.append({
        "round": k, "adopted": improved,
        "train_metrics": train_metrics,
        "candidate_dev_accuracy": cand_score,
        "candidate_dev_metrics": cand_dev,
        "best_dev_accuracy": best_dev_score,
        "candidate_prompt": candidate_prompt,
    })
 
    # Persist after every round so a crash never loses progress.
    with open(os.path.join(OUT_DIR, "tuning_history.json"), "w") as fp:
        json.dump(history, fp, indent=2)
    with open(os.path.join(OUT_DIR, "best_prompt.txt"), "w") as fp:
        fp.write(best_prompt)
 
print(f"\nDone. Best dev accuracy = {best_dev_score:.4f}")
print(f"Best prompt written to {OUT_DIR}/best_prompt.txt")
print("Adopted rounds:",
      [h["round"] for h in history if h.get("adopted")])
