# -*- coding: utf-8 -*-

import os
import sys
import tiktoken
import pandas as pd
import json


# Define the path to the dataset folder containing unpacked phishing data
ws_path = "/workspace/dataset/unpacked_folder_phishing"

# List all files/folders inside the dataset path
files = os.listdir(ws_path)

# Dictionary to store HTML content with folder names as keys
all_html1 = {}

# Iterate over each folder/file in the dataset path
for filename in files:
    # Construct the full path of the current item
    p2 = os.path.join(ws_path, filename).replace("\\", "/")
    
    # List files within the current folder
    wss = os.listdir(p2)
    
    # Iterate through files in the folder
    for f1 in wss:
        # Process only files ending with ".html"
        if f1.endswith(".html"):
            # Construct full path of the HTML file
            file_path = os.path.join(p2, f1).replace("\\", "/")
            
            # Open the HTML file in read mode with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as f:
                # Store the content in a dictionary with the folder name as key
                all_html1[filename] = f.read()
                
                
# Step 1: Initialize GPT-4 tokenizer using tiktoken
model_name = "gpt-4"
encoding = tiktoken.encoding_for_model(model_name)

# Step 2: Count tokens for each HTML document
doc_token_counts = {name: len(encoding.encode(doc)) for name, doc in all_html1.items()}

# Step 3: Convert token counts into a pandas DataFrame
df = pd.DataFrame(list(doc_token_counts.items()), columns=["doc_name", "token_count"])

# Step 4: Split documents into 10 bins based on token counts (quantile-based binning)
df["token_bin"] = pd.qcut(df["token_count"], q=10, labels=False)

# Step 5: Sample 500 documents in total, proportionally distributed across bins (50 per bin)
sampled_df = df.groupby("token_bin", group_keys=False).apply(
    lambda x: x.sample(n=int(500 / 10), random_state=42)  # random_state ensures reproducibility
)

# Step 6: Extract only the sampled document names
sampled_doc_names = sampled_df["doc_name"].tolist()
print(f"Number of sampled document names: {len(sampled_doc_names)}")


# Add custom script path for importing truncate functions
ROOT = "/workspace/scripts"
sys.path.append(ROOT)

# Import helper functions to truncate HTML by token length
from truncate_html_functions_github_version import *

# Dictionaries to store truncated versions of documents
p5 = {}   # Stores 5% of tokens
p50 = {}  # Stores 50% of tokens

# Iterate over all documents
for name, doc in all_html1.items():
    if name in sampled_doc_names:  # Only process sampled documents
        doc_len = len(encoding.encode(doc))  # Get token length of document
        
        # Truncate to 5% of original length
        p5[name] = truncate_html_to_tokens_merged(doc, doc_len*0.05)
        
        # Truncate to 50% of original length
        p50[name] = truncate_html_to_tokens_merged(doc, doc_len*0.50)
        
# Save the truncated 5% documents into JSON
with open('/workspace/dataset/temp/phish_5.json', 'w') as fp:
    json.dump(p5, fp)

# Save the truncated 50% documents into JSON
with open('/workspace/dataset/temp/phish_50.json', 'w') as fp:
    json.dump(p50, fp)

      

        
