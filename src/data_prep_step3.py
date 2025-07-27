#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data_prep_step3.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_and_save(input_jsonl: str, output_dir: str, max_len: int = 512):
    """
    1) Loads the JSONL at `input_jsonl` (expects one {"text": ...} per line).
    2) Tokenizes to GPT-2 input_ids (truncates/pads to max_len).
    3) Sets labels = input_ids for causal LM.
    4) Saves the processed dataset to `output_dir` (creates it if needed).
    """
    # a) Load the prompts as a Hugging Face Dataset
    ds = load_dataset("json", data_files=input_jsonl, split="train")

    # b) Load GPT-2 tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # c) Tokenization function (batched)
    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )
        # Use the inputs themselves as labels for next‐token prediction
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # d) Apply it
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    # e) Save to disk
    os.makedirs(output_dir, exist_ok=True)
    tokenized.save_to_disk(output_dir)
    print(f"> Saved {len(tokenized)} examples to '{output_dir}'")

if __name__ == "__main__":
    # Filepaths from Step 2
    full_in  = "enron_prompts_full.jsonl"
    subj_in  = "enron_prompts_subject.jsonl"

    # Output folders for Step 4’s training
    full_out = "tokenized_full"
    subj_out = "tokenized_subject"

    print("💬 Tokenizing full‐thread prompts …")
    tokenize_and_save(full_in, full_out)

    print("💬 Tokenizing subject+last‐email prompts …")
    tokenize_and_save(subj_in, subj_out)

    print("✅ Step 3 complete.")


# In[ ]:




