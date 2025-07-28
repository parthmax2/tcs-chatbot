---

title: GPT-2 TCS Chatbot
emoji: "🤖"
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: true
------------

# 🤖 TCS GPT‑2 Chatbot

A FastAPI‑based chatbot powered by a fine‑tuned GPT‑2 model trained on real‑world TCS‑related Q\&A. Built and deployed on **Hugging Face Spaces** with Docker for a lightweight, scalable solution.

---

## 🚀 Demo

👉 **[Live Chatbot](https://huggingface.co/spaces/parthmax/gpt2-tcs-chatbot)**
Ask TCS‑related questions and get AI‑generated answers instantly.

---

## 🧠 About the Model

This chatbot relies on a fine‑tuned version of **`gpt2`** trained on a custom dataset of Q\&A pairs covering:

* HR processes
* Technical rounds
* Project scenarios
* Corporate policies

Training leveraged PyTorch and 🤗 Transformers with parameter‑efficient tuning (LoRA) for speed and resource savings.

---

## 🛠 Tech Stack

| Layer          | Details                          |
| -------------- | -------------------------------- |
| **Model**      | GPT‑2 (fine‑tuned)               |
| **Backend**    | FastAPI + Uvicorn                |
| **Deployment** | Hugging Face Spaces (Docker SDK) |
| **Frontend**   | Static HTML served via FastAPI   |

---

## 🧩 Features

* Clean, fast REST API via FastAPI
* Heuristic fallback for unreliable answers
* Fully Dockerised for reproducible builds
* Easy to extend with new data or UI themes

---

## 📦 Local Usage

```bash
# Clone and enter the directory
git clone https://huggingface.co/spaces/parthmax/gpt2-tcs-chatbot
cd gpt2-tcs-chatbot

# Build and run with Docker
docker build -t tcs-chatbot .
docker run -p 7860:7860 tcs-chatbot
```

---

## 📁 File Overview

| File               | Purpose                               |
| ------------------ | ------------------------------------- |
| `app.py`           | FastAPI backend to serve the chatbot  |
| `index.html`       | Minimal frontend UI                   |
| `infer.py`         | CLI‑based inference script (optional) |
| `train.py`         | GPT‑2 fine‑tuning script              |
| `Dockerfile`       | Production Docker configuration       |
| `requirements.txt` | Python dependencies                   |

---

## 👤 Author

**Saksham Pathak** ([`@parthmax`](https://huggingface.co/parthmax))
GenAI & LLM Developer • Specialised in fine‑tuning, RAG, transformers

---

## 📜 License

MIT License

---

> 🔗 Powered by Hugging Face Transformers & Spaces
