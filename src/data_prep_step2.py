#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data_prep_step2.py

import json

def build_prompts(input_jsonl: str,
                  output_full: str,
                  output_subject: str):
    """
    Reads each example from input_jsonl (which has keys:
      'thread', 'subject', 'email', 'reply', 'tone')
    and writes two JSONL files, one per prompt format.
    """
    with open(input_jsonl, encoding='utf-8') as fin, \
         open(output_full,    'w', encoding='utf-8') as fout_full, \
         open(output_subject, 'w', encoding='utf-8') as fout_subj:

        for line in fin:
            ex = json.loads(line)

            # 1) Full-thread prompt
            prompt_full = (
                f"{ex['tone']} Thread: {ex['thread']} Reply: {ex['reply']}"
            )
            fout_full.write(
                json.dumps({"text": prompt_full}, ensure_ascii=False) + "\n"
            )

            # 2) Subject + last email prompt
            prompt_subj = (
                f"{ex['tone']} Subject: {ex['subject']} ┃ "
                f"Last message: {ex['email']} Reply: {ex['reply']}"
            )
            fout_subj.write(
                json.dumps({"text": prompt_subj}, ensure_ascii=False) + "\n"
            )

if __name__ == "__main__":
    input_jsonl    = "enron_pairs.jsonl"
    output_full    = "enron_prompts_full.jsonl"
    output_subject = "enron_prompts_subject.jsonl"

    print("Building prompts…")
    build_prompts(input_jsonl, output_full, output_subject)
    print("Done.")


# In[ ]:




