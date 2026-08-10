# AI Beginner Track (10 Weeks) — Builder Path

The companion track to `gen-ai/`. While `gen-ai/` teaches you to **use** GenAI tools, this track teaches you to **build with** AI — write code that runs models, builds RAG pipelines, and fine-tunes a model.

## Target Audience
- People who want to actually work with models, not just chat with them
- Basic familiarity with programming in any language (Python preferred)
- No advanced math or ML background required — we build intuition, not proofs

## Prerequisite
- Basic Python: lists, dicts, functions. That's it. If you can run a notebook, you're ready — Week 1 covers everything else you need.

## Learning Outcomes
By the end of this course, you can:
- Load and run pretrained models (inference) with Hugging Face
- Build a RAG pipeline that answers questions over your own documents
- Fine-tune a small model with LoRA on your own dataset
- Evaluate a model's output and know when it's actually better
- Deploy a simple model API

## Stack
- Python 3.10+, Hugging Face `transformers`, `datasets`, `peft`, PyTorch, Jupyter/Colab (free GPUs)

## Course Structure
- Duration: 10 weeks
- Format: Weekly lesson + lab + assignment
- Milestone: Build one real project by the end

---

## Week 1: Python for AI
- The Python you actually need: lists, dicts, functions, comprehensions
- JSON/JSONL data (the format all AI datasets use)
- Jupyter notebooks and Google Colab (free GPU)

## Week 2: How LLMs Work
- The core idea: next-token prediction
- Tokens, tokenizers, embeddings, attention
- Training vs inference — the two phases
- Lab: see tokens and next-token probabilities with your own eyes

## Week 3: Run Your First LLM
- Hugging Face `transformers`: `pipeline()` in 3 lines
- Text generation, sentiment, question-answering, zero-shot classification
- Model sizes and memory: what fits on your machine
- Lab: run and compare real models

## Week 4: Pretrained Models and Inference
- Task pipelines: text generation, sentiment, Q&A, zero-shot classification
- Generation settings: temperature, top-k, top-p, max tokens — what they actually do
- Model sizes and memory: what fits on your machine
- Lab: run and compare real models

## Week 5: The Inference Pipeline (Under the Hood)
- The full path: tokenize → forward pass → sample → decode
- Logits, probability distributions, and the auto-regressive loop
- Batching and GPU vs CPU
- Lab: build a minimal inference script without `pipeline()`

## Week 6: RAG — Retrieval-Augmented Generation
- Embeddings: turning text into vectors; similarity (cosine/dot product)
- Vector databases and similarity search (FAISS)
- Chunking and indexing your documents
- The full RAG loop: query → retrieve → prompt → answer
- Lab: Q&A bot over your own documents

## Week 7: Fine-Tuning with LoRA
- When to fine-tune vs use RAG vs just prompt
- How training works: the gradient-descent loop (appendix)
- Dataset formats for fine-tuning (instruction datasets)
- LoRA: what it is and why it's cheap
- Lab: fine-tune a small model (e.g., GPT-2 / SmolLM) on a dataset with free Colab GPU

## Week 8: Evaluation
- Why "it looks good" isn't a metric
- Eval sets, held-out data, and regression testing
- Basic metrics: accuracy, F1, BLEU, perplexity
- Lab: score your fine-tuned model vs the base model

## Week 9: Serving and Deployment
- Wrap a model in an API (FastAPI)
- Gradio for quick demos
- Cost/latency basics: quantization, GPU memory
- Lab: deploy your model as a public demo

## Week 10: Capstone
- Pick a real problem, build a full pipeline (RAG or fine-tune or both)
- Document it in a GitHub repo with a README
- Present before/after impact

---

## Milestones
- Weeks 1–3: The base — Python for AI, how LLMs work, first inference
- Weeks 4–5: The machinery — inference pipeline, embeddings
- Weeks 6–7: Build — RAG + fine-tuning
- Weeks 8–10: Validate and ship — eval, serving, capstone

## Assessment
- Weekly assignments: 60%
- Labs: 20%
- Capstone: 20%

## Required Resources
- Google Colab (free) or any Python environment
- Hugging Face account (free, for model access)
- GitHub repo for labs and the capstone

---

## Why This Order
This follows the industry-standard HF learning path: **Python → what a model is → run pretrained models → build with them (RAG) → adapt them (fine-tuning) → evaluate → deploy**. We deliberately defer the deep math (training from scratch, RLHF, quantization internals) to a future advanced track — here we build working systems and learn the concepts on the way.
