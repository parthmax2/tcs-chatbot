import argparse
import re
import torch
from collections import Counter
from transformers import AutoTokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="./gpt2_tcs/epoch-10")
parser.add_argument("--max_new_tokens", type=int, default=40)
parser.add_argument("--beam_width", type=int, default=4)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=1.2)
parser.add_argument("--min_words", type=int, default=3)
parser.add_argument("--uniq_ratio", type=float, default=0.4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
model = GPT2LMHeadModel.from_pretrained(args.ckpt).to(device)
model.eval()

print("=" * 60)
print(f"Loaded fine‑tuned GPT‑2 from → {args.ckpt}")
print("Ask your TCS questions (Ctrl‑C to quit)")
print("=" * 60)

fallback_msg = "I’m sorry, I’m not sure about that TCS topic yet."

def looks_unreliable(ans: str) -> bool:
    words = ans.split()
    if len(words) < args.min_words:
        return True
    uniq_ratio = len(set(words)) / len(words)
    if uniq_ratio < args.uniq_ratio:
        return True
    if re.search(r'\bvariation\b', ans, re.I):
        return True
    most_common = Counter(words).most_common(1)[0][1]
    if most_common / len(words) > 0.5:
        return True
    return False

while True:
    try:
        user_q = input("\nYou: ").strip()
        if not user_q:
            continue

        prompt = f"{tokenizer.bos_token} Question: {user_q} Answer:"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if args.sample:
            gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        else:
            gen_kwargs.update(num_beams=args.beam_width, early_stopping=True)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = decoded.split("Answer:", 1)[-1].strip()

        if looks_unreliable(answer):
            print(f"Bot: {fallback_msg}")
        else:
            print(f"Bot: {answer}")

    except KeyboardInterrupt:
        print("\nBye!")
        break
