# Week 3: Run Your First LLM

## Core Concepts

### The Gift You're Standing On
You don't train an LLM — you **download one** and run it. Pretrained models (GPT-2, SmolLM2, Qwen, Llama) are published with weights that cost millions to produce. You inherit all of that in three lines of code. This is *inference*: the weights are frozen, only the math runs forward.

### Hugging Face — the standard library
| Library | Purpose |
|---------|---------|
| `transformers` | Load and run any model |
| `datasets` | Download/process datasets |
| `peft` | Cheap fine-tuning (LoRA) — Week 7 |

### Install
```bash
pip install transformers torch
```

### Your First Model — 3 Lines
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("The capital of France is", max_new_tokens=10)[0]["generated_text"])
```

That's it. You just ran a language model.

### Task Pipelines (one-line superpowers)
```python
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
classifier("I love this course!")      # [{'label': 'POSITIVE', 'score': 0.99...}]

qa = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
ctx = "Hugging Face is a company that builds AI tools for developers."
qa(question="Who is Hugging Face?", context=ctx)

classify = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classify("The stock market rallied today", candidate_labels=["business", "politics", "sports"])
```

| Task | pipeline name |
|------|---------------|
| Continue/generate text | `text-generation` |
| Positive/negative | `sentiment-analysis` |
| Extract answer from a passage | `question-answering` |
| Classify into any labels | `zero-shot-classification` |
| Get embeddings (Week 5) | `feature-extraction` |
| Summarize / translate | `summarization` / `translation` |

### What Actually Happened
1. The library **downloaded the weights** to your disk (~500 MB for GPT-2, ~250 MB for DistilBERT). One-time, cached.
2. Your text was **tokenized** into IDs (Week 2).
3. The model ran **forward** — the same matrix math you saw in Week 2 — producing next-token scores.
4. It **sampled** a token, appended it, and repeated until done.

That loop is inference. No gradient descent here — that's training, which you'll do in Week 7.

### Models Come in Sizes
| Model | Params | Size on disk | What it's good for |
|-------|--------|-------------|--------------------|
| DistilBERT | 66M | ~250 MB | Classification, embeddings |
| GPT-2 | 124M | ~500 MB | Text generation (small) |
| SmolLM2-135M | 135M | ~0.5 GB | Small chat/generation — your course workhorse |
| Qwen-2.5-1.5B | 1.5B | ~3 GB | Better quality, needs GPU |
| Llama-3-8B | 8B | ~16 GB | Real chat (needs strong GPU or quantized) |

**Rule of thumb:** ~2 bytes per parameter at 16-bit. A 1.5B model ≈ 3 GB. Free Colab GPU handles up to ~7B.

---

## Key Takeaway
Running an LLM is a 3-line exercise with `transformers`. The model downloads once, then inference is: tokenize → forward → sample → decode, repeated. Sizes dictate what fits your hardware — there's always a model small enough for a laptop.

---

## Common Pitfalls

### Pitfall 1: Downloading a model too big for your machine
**Fix:** Check the model card for memory. Start small (SmolLM2-135M) and scale up.

### Pitfall 2: Slow first run
The download happens once and is cached. On Colab it re-downloads per session — normal.

### Pitfall 3: Wrong task name
`pipeline("summarize")` fails — it's `"summarization"`. Check the docs.

### Pitfall 4: Ignoring the device
CPU is slower but fine for small models. On GPU, pass `device=0` for ~10–50× speedup.

---

## Interactive Exercises

1. Run all four pipelines above. Record one surprising output each.
2. Run GPT-2/SmolLM2 on your own seed phrase. Notice how it drifts from the prompt — the model is wandering, not "understanding."
3. Change `max_new_tokens` from 10 to 100. What changes? Why?
4. Ask the Q&A model a question whose answer is *not* in the context. What does it do?
5. Run the same seed through two different models and compare.

---

## Quick Reference

```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2", device=0)   # GPU
gen("Once upon a time", max_new_tokens=50, do_sample=True, temperature=0.8)
```

---

## Further Reading
- Hugging Face course (free, canonical — start here): https://huggingface.co/learn/nlp-course
- SmolLM2 (your workhorse model): https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
