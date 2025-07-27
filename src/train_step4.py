#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# train_step4_v2.py

import os
import logging
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fine_tune(tokenized_dir, output_dir):
    # 1) Load and split
    ds = load_from_disk(tokenized_dir)
    ds = ds.train_test_split(test_size=0.1) if not isinstance(ds, dict) else ds
    train_ds = ds.get("train", ds.get("all"))
    eval_ds  = ds.get("test",  ds.get("validation"))

    logger.info(f"→ {tokenized_dir}: train={len(train_ds)}, eval={len(eval_ds)}")

    # 2) Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 3) Inject LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_cfg)

    # 4) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) Train & save
    logger.info(f"✨ Fine‐tuning {tokenized_dir} → {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"✔ Done {output_dir}\n")

if __name__ == "__main__":
    pairs = [
        ("tokenized_full",    "outputs/gpt2_lora_full"),
        ("tokenized_subject", "outputs/gpt2_lora_subject"),
    ]
    for tok_dir, out_dir in pairs:
        os.makedirs(out_dir, exist_ok=True)
        fine_tune(tok_dir, out_dir)


# In[ ]:




