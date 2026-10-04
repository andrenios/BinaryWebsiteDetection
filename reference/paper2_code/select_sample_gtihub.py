# -*- coding: utf-8 -*-


import pandas as pd
import zipfile
from PIL import Image
import numpy as np
from io import BytesIO
import tiktoken

nophish = pd.read_csv("", sep = ",")
nophish["language"].value_counts()

yesphish = pd.read_csv("", sep = ",")
yesphish["language"].value_counts()

# Select Sample

# 1) Tag websites with all black or all white screenshot

img_black_whit_no_phish = []

with zipfile.ZipFile("", "r") as f:
    names = f.namelist()
    images = [f for f in names if f.endswith(".jpg")]
    
    for img in images:
        with f.open(img) as file:
            jpg = Image.open(BytesIO(file.read())).convert("L")  # grayscale
            arr = np.array(jpg)
            
            if np.all(arr == 0) or np.all(arr == 255):
                img_black_whit_no_phish.append(img.split("/")[1])
        
        
img_black_white_phish = []

with zipfile.ZipFile("", "r") as f:
    names = f.namelist()
    images = [f for f in names if f.endswith(".jpg")]
    
    for img in images:
        with f.open(img) as file:
            jpg = Image.open(BytesIO(file.read())).convert("L")  # grayscale
            arr = np.array(jpg)
            
            if np.all(arr == 0) or np.all(arr == 255):
                img_black_white_phish.append(img.split("/")[1])        

# 1.1) Filter csv
nophish_filt = nophish[~nophish["_id"].isin(img_black_whit_no_phish)]
yesphish_filt = yesphish[~yesphish["_id"].isin(img_black_white_phish)]

nophish_filt["language"].value_counts()
yesphish_filt["language"].value_counts()



# 2) Create dataset for stratified sampling

encoding = tiktoken.encoding_for_model("gpt-4")

# No Phishing

_id = nophish_filt["_id"]
language = nophish_filt["language"]

html_len_no_phish = []

with zipfile.ZipFile("", "r") as f:
    names = f.namelist()
    htmls = [fi for fi in names if fi.endswith("html") and fi.split("/")[1] not in img_black_whit_no_phish]

    for hfile in htmls:
        with open(hfile, "r", encoding="utf-8", errors="ignore") as f1:
            html = f1.read()
            html_len_no_phish.append(len(encoding.encode(str(html))))
            
nophish_sampling = pd.DataFrame({"id": _id, "language": language, "html_length": html_len_no_phish})


# Phishing

_id = yesphish_filt["_id"]
language = yesphish_filt["language"]

html_len_phish = []

with zipfile.ZipFile("", "r") as f:
    names = f.namelist()
    htmls = [fi for fi in names if fi.endswith("html") and fi.split("/")[1] not in img_black_white_phish]

    for hfile in htmls:
        with open(hfile, "r", encoding="utf-8", errors="ignore") as f1:
            html = f1.read()
            html_len_phish.append(len(encoding.encode(str(html))))
            
phish_sampling = pd.DataFrame({"id": _id, "language": language, "html_length": html_len_phish})


# 3) Sample Data
def stratified_split(df, test_frac=0.30, n_length_buckets=3, seed=42, verbose=True):
    """Return (train_df, test_df), stratified on language x length bucket.
 
    Parameters
    ----------
    df : DataFrame with columns id, language, html_length
    test_frac : fraction going to the test/quarantine set
    n_length_buckets : how many html_length bins to stratify on (quantile-based)
    seed : random seed for reproducibility
    verbose : print the resulting stratum sizes
    """
    df = df.copy()
 
    # Bucket the continuous length into quantile bins so it can be stratified on.
    df["_len_bucket"] = pd.qcut(df["html_length"], q=n_length_buckets,
                                labels=False, duplicates="drop")
 
    # Within each (language, length-bucket) stratum, sample test_frac to test.
    test_parts = []
    for _, group in df.groupby(["language", "_len_bucket"]):
        test_parts.append(group.sample(frac=test_frac, random_state=seed))
 
    test_df = pd.concat(test_parts)
    train_df = df.drop(test_df.index)
 
    # Drop the helper column before returning.
    train_df = train_df.drop(columns="_len_bucket")
    test_df = test_df.drop(columns="_len_bucket")
 
    if verbose:
        print(f"train: {len(train_df):,}  test: {len(test_df):,}  "
              f"(test = {len(test_df) / len(df):.1%})")
        print("\nlanguage balance:")
        print(pd.DataFrame({
            "train": train_df["language"].value_counts(normalize=True).round(3),
            "test": test_df["language"].value_counts(normalize=True).round(3),
        }))
 
    return train_df, test_df

ben_train, ben_test = stratified_split(nophish_sampling)
phish_train, phish_test = stratified_split(phish_sampling)
