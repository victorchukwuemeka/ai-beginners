# Week 9 Assignment: Deploy and Document a Model Service

## Due
End of Week 9

## Objective
Ship a documented, working model service — not a notebook — and write down its performance characteristics.

## Requirements

### 1. A real FastAPI service (`app.py`)
- `POST /generate` with input validation (`prompt` required, `max_new_tokens` capped)
- Clean JSON responses, no crashes on bad input
- Loads the model once at startup (not per-request)
- Runs locally with uvicorn

### 2. Model decision memo (`SERVING.md`)
For the model you chose to serve, write:
- Why this model (size vs task fit)
- Memory footprint (fp16 vs 4-bit) with the actual numbers from your lab
- Latency measured on 10 requests (avg seconds/request)
- What you'd do for more users: batch? quantize? smaller model? bigger GPU? (one paragraph)

### 3. Demo
A Gradio demo (local or `share=True`) with a screenshot or URL. If you fine-tuned in Week 7, serve the fine-tuned model here.

### 4. Cost sanity check (quick math)
Estimate serving cost: `avg_seconds × requests_per_day ÷ 3600 × GPU_hourly_rate`. Use a real-ish rate (e.g., Colab ≈ free, a T4 cloud GPU ≈ $0.30–0.60/hr). Show the math. Is it viable?

### 5. Reflection (150–300 words)
- What was the hardest part of going from notebook to endpoint?
- What would break first under real traffic?

## Grading
| Criterion | Weight |
|-----------|--------|
| FastAPI service works, validates, handles errors | 30% |
| Serving memo with real numbers (memory, latency) | 25% |
| Gradio demo works | 15% |
| Cost math shown and interpreted | 15% |
| Reflection is specific | 15% |
