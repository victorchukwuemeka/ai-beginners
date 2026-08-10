# Week 9: Serving and Deployment

## Core Concepts

### The Gap Between Notebook and Product
Your model works in a notebook. A product needs it accessible to other programs and people: an **API endpoint** that takes a request and returns an answer, cheaply and reliably. That transition is "serving."

### Option 1: Gradio (fastest demo)
```bash
pip install gradio
```

```python
import gradio as gr
from transformers import pipeline

gen = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")

def chat(message, history):
    return gen(message, max_new_tokens=100)[0]["generated_text"]

gr.ChatInterface(chat).launch()
```
One file → a shareable URL (Gradio `launch(share=True)` gives a public link). Perfect for demos and showing non-technical people.

### Option 2: FastAPI (real API)
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
gen  = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 100

@app.post("/generate")
def generate(q: Query):
    return {"output": gen(q.prompt, max_new_tokens=q.max_new_tokens)[0]["generated_text"]}
```
Run: `uvicorn app:app --reload`. Now ANY system can call `POST /generate`.

### Memory and Cost: Why Size Matters

**~2 bytes per parameter at 16-bit precision:**

| Model | Params | VRAM (fp16) | Realistic hardware |
|-------|--------|-------------|--------------------|
| SmolLM2-135M | 135M | ~0.3 GB | Any laptop |
| Qwen-2.5-1.5B | 1.5B | ~3 GB | Laptop GPU / mid GPU |
| Llama-3-8B | 8B | ~16 GB | A100 / RTX 4090-ish |
| Llama-3-70B | 70B | ~140 GB | Multi-GPU — you won't serve this alone |

### Quantization: Shrink the Model
Quantization stores weights in fewer bits — 4-bit cuts memory ~4× with small quality loss. This is why people serve 7B models on a single GPU.

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", quantization_config=bnb)
```

### Batch and Cache to Cut Cost
- **Batching** — process multiple requests per forward pass → more throughput per GPU.
- **KV cache** — the model caches earlier computations per request so each new token doesn't recompute everything. `generate()` does this for you.

### Latency / Cost / Quality Trade-off
Serving is a triangle: faster + cheaper usually means smaller/quantized = slightly lower quality. Decide by task:
- Chatbot for customers → speed + cost matter, quality still high → small model + RAG
- Offline batch analysis → size matters less

---

## Key Takeaway
Serving = wrap your model in an endpoint (Gradio for demos, FastAPI for real integrations). Model size drives memory and cost; quantization (4-bit) and batching keep it manageable. Deploy small and measure before scaling up.

---

## Common Pitfalls

### Pitfall 1: Serving a model too big for the box
OOM crash at startup = model doesn't fit. Check VRAM budget, quantize, or pick a smaller model.

### Pitfall 2: Cold-start re-download
Server downloads the model on first request → slow first call. Preload at import, not inside the endpoint.

### Pitfall 3: Unbounded request size
A request with a 10,000-token prompt on a small GPU will time out. Cap `max_new_tokens` and input length.

### Pitfall 4: No error handling
Model dies on a malformed input → 500 error. Wrap in try/except and return a clean message.

### Pitfall 5: Forgetting it's now public
If you `share=True`, anyone with the link can hit your model. Watch for abuse; don't expose secrets.

---

## Interactive Exercises

1. Build the Gradio ChatInterface and launch with `share=True`. Send the link to someone.
2. Build the FastAPI app. Test it with `curl -X POST .../generate -H "Content-Type: application/json" -d '{"prompt":"hello"}'`.
3. Check memory: load the same model at fp16 vs 4-bit; compare `model.get_memory_footprint()`.
4. Load Qwen-2.5-1.5B if it fits and compare quality vs SmolLM2.
5. Add a max-tokens cap and see how it protects a low-memory machine.

---

## Quick Reference

```bash
pip install fastapi uvicorn gradio
```

```python
# FastAPI endpoint pattern
@app.post("/generate")
def generate(q: Query):
    try:
        out = gen(q.prompt, max_new_tokens=min(q.max_new_tokens, 512))
        return {"output": out[0]["generated_text"]}
    except Exception as e:
        return {"error": str(e)}, 500
```

---

## Further Reading
- FastAPI docs: https://fastapi.tiangolo.com
- Hugging Face inference guide: https://huggingface.co/docs/transformers/en/main_classes/pipelines
- vLLM (production-grade serving, later step): https://github.com/vllm-project/vllm
