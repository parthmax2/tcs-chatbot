import os
import re
from collections import Counter

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, GPT2LMHeadModel

# ====================== Config ======================
CKPT_PATH = os.getenv("CKPT_PATH", "./gpt2_tcs/epoch-10")  # or replace with your HF model name
MAX_NEW_TOKENS = 40
BEAM_WIDTH = 4
REPETITION_PENALTY = 1.2
MIN_WORDS = 3
UNIQ_RATIO = 0.4
FALLBACK_MSG = "I’m sorry, I’m not sure about that TCS topic yet."

# ====================== Model Setup ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
model = GPT2LMHeadModel.from_pretrained(CKPT_PATH).to(device)
model.eval()

def looks_unreliable(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_WORDS:
        return True
    if len(set(words)) / len(words) < UNIQ_RATIO:
        return True
    if re.search(r"\bvariation\b", text, re.I):
        return True
    if Counter(words).most_common(1)[0][1] / len(words) > 0.5:
        return True
    return False

@torch.inference_mode()
def generate_answer(question: str) -> str:
    prompt = f"{tokenizer.bos_token} Question: {question.strip()} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=BEAM_WIDTH,
        early_stopping=True,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(ids[0], skip_special_tokens=True).split("Answer:", 1)[-1].strip()
    return FALLBACK_MSG if looks_unreliable(answer) else answer

# ====================== FastAPI App ======================
app = FastAPI(title="TCS Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return {"answer": generate_answer(req.question)}

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
