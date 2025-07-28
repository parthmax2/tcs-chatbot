"""
train.py – Fine‑tune GPT‑2 on local TCS Q&A data
Compatible with Python 3.13 & CUDA 12.6 (PyTorch 2.7.1)
"""

import os
import json
import importlib
import re

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

NUM_EPOCHS = 10
BATCH_SIZE = 8
LR = 5e-5
WDECAY = 0.01
WARMUP_RATIO = 0.06
MAX_LEN = 768
SAVE_DIR = "./gpt2_tcs"
JSON_PATH = "data/tcs_data2.json"
PATIENCE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device → {device}")

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

try:
    importlib.import_module("triton")
    model = torch.compile(model)
    print("✓ torch.compile enabled")
except Exception:
    print("⚠ Triton unavailable – continuing without torch.compile")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_data = json.load(f)

ds = Dataset.from_list(json_data)

variation_re = re.compile(r"\s*\(variation\s*\d+\)\s*$", re.IGNORECASE)

def preprocess(example):
    clean_q = variation_re.sub("", example["question"]).strip()
    text = (
        f"{tokenizer.bos_token}"
        f" Question: {clean_q}"
        f" Answer: {example['answer']}"
        f" {tokenizer.eos_token}"
    )
    return tokenizer(text, add_special_tokens=False, truncation=True, max_length=MAX_LEN)

ds = ds.map(preprocess, remove_columns=["question", "answer"])
ds.set_format(type="torch")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

dl = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collator, pin_memory=True)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WDECAY)
total_steps = NUM_EPOCHS * len(dl)
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

scaler = GradScaler()

progress = tqdm(range(total_steps), desc="Training", unit="step")
best_loss = float("inf")
no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for batch in dl:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16, device_type="cuda"):
            loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss

        running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        progress.update(1)

    avg_loss = running_loss / len(dl)
    print(f"Epoch {epoch}: avg training loss = {avg_loss:.4f}")

    ckpt_dir = os.path.join(SAVE_DIR, f"epoch-{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    if avg_loss + 1e-4 >= best_loss:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stop: no improvement.")
            break
    else:
        best_loss = avg_loss
        no_improve = 0

print(" Training complete.")