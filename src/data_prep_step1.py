#!/usr/bin/env python
# coding: utf-8

# """
# fine_tune_gpt2_lora.py
# 
# Fine-tune GPT-2 Small on Enron email–reply pairs with LoRA adapters,
# using either full-thread or subject+last-email prompts and tone conditioning.
# """
# need to test the pipeline from local to GitHub

# In[ ]:


pip install transformers datasets peft accelerate


# In[ ]:


import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# Data Preparation

# In[ ]:


# data_prep_step1.py

import pandas as pd
import json
import re
import warnings

def extract_tag_index(tag: str) -> int:
    """
    Map message tags to an integer index for ordering:
      "<|original|>" → 0
      "<|replyN|>"   → N
    Anything else → large number (sorts last).
    """
    if tag == "<|original|>":
        return 0
    m = re.match(r"<\|reply(\d+)\|>", tag)
    if m:
        return int(m.group(1))
    return 9999

def prepare_pairs(input_csv: str, output_jsonl: str):
    # 1) Load the CSV (local or URL), skipping malformed lines
    csv_kwargs = {
        "engine": "python",
        "on_bad_lines": "skip",
        "encoding": "utf-8"
    }
    if urlparse(input_csv).scheme in ("http", "https"):
        df = pd.read_csv(input_csv, **csv_kwargs)
    else:
        df = pd.read_csv(input_csv, **csv_kwargs)


#def prepare_pairs(input_csv: str, output_jsonl: str):
    # 1) Load the CSV, skipping malformed lines
   # with warnings.catch_warnings():
   #     warnings.simplefilter("ignore")
   #     df = pd.read_csv(
    #        input_csv,
    #        engine='python',
    #        on_bad_lines='skip'
     #   )

    # 2) Ensure there's a 'subject' column (blank for now)
    if 'subject' not in df.columns:
        df['subject'] = ""

    examples = []

    # 3) Process each thread
    for thread_id, group in df.groupby("message_id", sort=False):
        grp = group.copy()
        # 4) Sort messages by tag index
        grp['idx'] = grp['tag'].apply(extract_tag_index)
        grp = grp.sort_values('idx')

        msgs = grp['clean_message'].tolist()
        # 5a) Grab the subject from the original message row
        subject_rows = grp.loc[grp['idx'] == 0, 'subject']
        subject = subject_rows.iloc[0] if not subject_rows.empty else ""

        # 5b) For each reply position i (i >= 1), build one example
        for i in range(1, len(msgs)):
            examples.append({
                "thread": " ".join(msgs[:i]),  # all messages before reply i
                "subject": subject,            # blank or to-be-filled later
                "email": msgs[i-1],            # immediate predecessor
                "reply": msgs[i],              # this reply
                "tone": "[formal]"             # placeholder tone
            })

    # 6) Write examples to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for ex in examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

#if __name__ == "__main__":
#    input_csv    = "enron_cleaned_v7.csv"
#    output_jsonl = "enron_pairs.jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Build email→reply pairs from a CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/sample_raw_dataset.csv",
        help="Path or URL to the raw CSV (default: data/sample_raw_dataset.csv)"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="enron_pairs.jsonl",
        help="Where to write the output JSONL (default: enron_pairs.jsonl)"
    )
    args = parser.parse_args()
    

    print(f"Reading from: {args.input_csv}")
    prepare_pairs(args.input_csv, args.output_jsonl)
    count = sum(1 for _ in open(args.output_jsonl, encoding="utf-8"))
    print(f"Wrote {count} examples to {args.output_jsonl}")


# In[14]:


# Move up from src/ into the project root
get_ipython().run_line_magic('cd', '../')


# In[13]:


input_csv   = "../data/sample_raw_dataset.csv"
output_json = "../data/sample_pairs.jsonl"


# In[2]:


#!/usr/bin/env python
# src/data_prep_step1.py

import argparse
import json
import pandas as pd
import re
import warnings
from urllib.parse import urlparse

def extract_tag_index(tag: str) -> int:
    """
    Map message tags to an integer index for ordering:
      "<|original|>" → 0
      "<|replyN|>"   → N
    Anything else → large number (sorts last).
    """
    if tag == "<|original|>":
        return 0
    m = re.match(r"<\|reply(\d+)\|>", tag)
    if m:
        return int(m.group(1))
    return 9999

def prepare_pairs(input_csv: str, output_jsonl: str):
    # 1) Load the CSV (local path or HTTP URL), skipping malformed lines
    csv_kwargs = {
        "engine": "python",
        "on_bad_lines": "skip",
        "encoding": "utf-8",
    }
    if urlparse(input_csv).scheme in ("http", "https"):
        df = pd.read_csv(input_csv, **csv_kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(input_csv, **csv_kwargs)

    # 2) Ensure 'subject' exists
    if "subject" not in df.columns:
        df["subject"] = ""

    examples = []

    # 3) Group by thread and build pairs
    for thread_id, group in df.groupby("message_id", sort=False):
        grp = group.copy()
        grp["idx"] = grp["tag"].apply(extract_tag_index)
        grp = grp.sort_values("idx")

        msgs = grp["clean_message"].tolist()
        # Grab subject from the original (idx == 0)
        orig_subj = grp.loc[grp["idx"] == 0, "subject"]
        subject = orig_subj.iloc[0] if not orig_subj.empty else ""

        # For each reply (i >= 1), form one example
        for i in range(1, len(msgs)):
            examples.append({
                "thread":  " ".join(msgs[:i]),   # all earlier messages
                "subject": subject,
                "email":   msgs[i - 1],          # last message before reply
                "reply":   msgs[i],              # the reply itself
                "tone":    "[formal]",           # placeholder tone
            })

    # 4) Write to JSONL
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for ex in examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Build email→reply pairs from a CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/sample_raw_dataset.csv",
        help="Path or URL to the raw CSV (default: data/sample_raw_dataset.csv)",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="enron_pairs.jsonl",
        help="Where to write the output JSONL (default: enron_pairs.jsonl)",
    )

    # Use parse_known_args so extra Jupyter flags are ignored
    args, _ = parser.parse_known_args()

    print(f"Reading from: {args.input_csv!r}")
    prepare_pairs(args.input_csv, args.output_jsonl)
    count = sum(1 for _ in open(args.output_jsonl, encoding="utf-8"))
    print(f"Wrote {count} examples to {args.output_jsonl!r}")



# In[ ]:




