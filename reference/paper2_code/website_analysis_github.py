import os
import sys
import ollama
import pandas as pd
import re
import json
import time
import ast
from PIL import Image

# Add project scripts to Python path
ROOT = ""
sys.path.append(ROOT)

# Import local helper functions and prompts
from local_llm_infer_gtihub import *
from prompt_tuned_text_github import *

# ---- Experiment mode ----
# "truncated" : iterate test.csv, read <id>.json summaries from test_json/
# "full"      : iterate test_og.csv, read index.html from test_og/<id>/
MODE = "truncated"
 
# ---- Paths ----
if MODE == "truncated":
    order_csv   = ""
    data_folder = ""
elif MODE == "full":
    order_csv   = ""
    data_folder = ""
else:
    raise ValueError(f"unknown MODE: {MODE}")
 
# ---- Read order + labels ----
# test.csv / test_og.csv columns: id, label, filename  (filename only in truncated)
order_df = pd.read_csv(order_csv)
 
 
def load_site(site_id, mode, data_folder):
    """Return the prompt input string for one site.
    truncated -> the JSON summary, pretty-printed to keep its structure.
    full      -> the raw index.html text."""
    if mode == "truncated":
        json_path = os.path.join(data_folder, f"{site_id}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return json.dumps(summary, indent=2, ensure_ascii=False)
    else:  # full
        html_path = os.path.join(data_folder, str(site_id), "index.html")
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

not_phish_all = pd.read_csv("")
phish_all = pd.read_csv("")

# ---- Read Models ----	
# Retrieve list of available models from Ollama
#model_list = [i["model"] for i in ollama.list()["models"]]
# Alternative: manually specify models
model_list = ["qwen3.6:35b", "qwen3.5:9b", "mistral-small3.2:24b", "gemma4:12b", "gemma4:31b", "muse-glimmer:30b"]
#model_list = ["gemma3:27b"]

# ---- Create JSON Dataset Loop ----
page_runs = 1  # number of runs per page for consistency checking

t5 = time.time()
for m in model_list:
    result_collection = {}  # stores results per model
    result_raw = {}         # placeholder for raw results (unused)

    if not m in result_collection.keys():
        result_collection[m] = {}
    
    print(m)
    print(time.time())
    
    pageID = 0  # unique ID per webpage

    # Iterate over dataset entries in the order given by the CSV
    for _, row in order_df.iterrows():
        ws_name = row["_id"]
        true_label = bool(row["label"])  # label comes from the CSV
        #print(ws_name)
        #print(true_label)
        #if true_label == False and MODE == "full":
        #    ws_url = not_phish_all[not_phish_all["_id"] == ws_name]["url"].to_list()[0]
        #elif true_label == True and MODE == "full":
        #    ws_url = phish_all[phish_all["_id"] == ws_name]["url"].to_list()[0]
            

    # Iterate over dataset entries (websites)
    #for ws_name, ws in list(d5.items()):
        #if pageID == 801:
        #    print(ws_name)
        #if pageID > 20:
        #    break
        
        # Initialize results for this page
        if not pageID in result_collection[m].keys():
            result_collection[m][pageID] = {}

        # Load the site's prompt input (JSON summary or raw HTML)
        html_summary = load_site(ws_name, MODE, data_folder)
        #print(len(html_summary))
        
        # Locate corresponding folder (benign or phishing)
        if true_label == True:
            image_folder = ""
        else:
            image_folder = ""
        
        target_folder = os.path.join(image_folder, ws_name)
        
        # Open screenshot image (assumed .jpg in folder)
        png_file = [f for f in os.listdir(target_folder) if f.lower().endswith(".jpg")][0]
        image_path = os.path.join(target_folder, png_file)
        
        # Extract HTML, URL, and label
        #ws_html = ws["html"]
        #ws_url = ws["url"]
        #true_label = ws_name in phish  # True if phishing
        
        # Run inference multiple times for this page
        #for i in range(page_runs):
        run_result = {}
        
        # Build prompt from HTML and metadata
        #html_prompt = build_html_prompt_v6(html_summary, len(html_summary), ws_url)
        html_prompt = build_html_prompt_v8(html_summary)
        
        # Run model inference and time it
        t1 = time.time()
        #analysis_result = local_llm_infer_v2(html_prompt, max_tokens=750, model=m)
        analysis_result = local_llm_infer_v3(html_prompt, image_path, max_tokens=750, model=m)
        t2 = time.time()
        
        # Store results
        run_result["ws_name"] = ws_name
        run_result["True_Phish_Label"] = true_label
        run_result["runtime"] = t2 - t1
        run_result["prompt_len"] = len(html_prompt)
        run_result["analysis_result"] = analysis_result
        
        result_collection[m][pageID] = run_result 

        # Save intermediate results periodically
        if pageID in [300, 600, 900, 1200, 1500, 1800]:
            save_loc1 = f"/res_{m}_{pageID}.json"
            with open(save_loc1, 'w') as fp:
                json.dump(result_collection, fp) 

        pageID += 1

    # Save results for this model
    save_loc = f"/res_{m}.json"
    with open(save_loc, 'w') as fp:
        json.dump(result_collection, fp) 

t6 = time.time()
