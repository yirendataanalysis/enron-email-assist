#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# evaluate_step5.py
#Load your fine-tuned GPT-2+LoRA model and tokenizer.
#Generate replies for each example using your chosen prompt format.
#Compute automatic metrics:
#Save a JSON report of all metrics.

import json
import math
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bertscore


def build_prompt(ex, fmt: str):
    """Construct the generation prompt (without the gold reply)."""
    tone = ex["tone"]
    if fmt == "full_thread":
        return f"{tone} Thread: {ex['thread']} Reply:"
    else:
        return (f"{tone} Subject: {ex['subject']} ┃ Last message: "
                f"{ex['email']} Reply:")


def compute_perplexity(model, tokenizer, texts, device):
    """Compute perplexity over a list of full <prompt + ' ' + reply> strings."""
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        neg_log_likelihood = outputs.loss * encodings["input_ids"].size(1)
    ppl = torch.exp(outputs.loss).item()
    return ppl


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",   required=True,
                   help="Path to fine-tuned model (e.g. outputs/gpt2_lora_full)")
    p.add_argument("--pairs_jsonl", default="enron_pairs.jsonl")
    p.add_argument("--prompt_format", choices=["full_thread", "subject_last_email"],
                   default="full_thread")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_gen_length", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # 2) Read pairs JSONL
    raw = [json.loads(l) for l in open(args.pairs_jsonl, encoding="utf-8")]

    # 3) Split off a validation slice (e.g., last 10%)
    n = len(raw)
    val = raw[int(0.9 * n):]

    # 4) Generate replies and collect references
    hyps, refs, ppl_texts = [], [], []
    for ex in tqdm(val, desc="Generating"):
        prompt = build_prompt(ex, args.prompt_format)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=args.max_gen_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        gen = tokenizer.decode(out_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
        hyps.append(gen.strip().split())
        refs.append([ex["reply"].strip().split()])
        ppl_texts.append(prompt + " " + ex["reply"].strip())

    # 5) Compute Perplexity
    print("📊 Computing perplexity…")
    perplexity = compute_perplexity(model, tokenizer, ppl_texts, device)

    # 6) BLEU
    print("📊 Computing BLEU…")
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie)

    # 7) ROUGE-L
    print("📊 Computing ROUGE-L…")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [scorer.score(" ".join(r[0]), " ".join(h))["rougeL"].fmeasure
                      for h, r in zip(hyps, refs)]
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    # 8) BERTScore
    print("📊 Computing BERTScore…")
    # We need untokenized strings again:
    hyp_strs = [" ".join(h) for h in hyps]
    ref_strs = [" ".join(r[0]) for r in refs]
    P, R, F1 = bertscore(hyp_strs, ref_strs, lang="en", rescale_with_baseline=True)
    bert_p, bert_r, bert_f1 = P.mean().item(), R.mean().item(), F1.mean().item()

    # 9) Report
    report = {
        "perplexity": perplexity,
        "bleu": bleu,
        "rouge_l": rouge_l,
        "bertscore": {"precision": bert_p, "recall": bert_r, "f1": bert_f1},
        "n_examples": len(val),
    }

    print("\n=== Evaluation Results ===")
    print(json.dumps(report, indent=2))
    with open("evaluation_report.json", "w") as fout:
        json.dump(report, fout, indent=2)
    print("Saved report → evaluation_report.json")


if __name__ == "__main__":
    main()

