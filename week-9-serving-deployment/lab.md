# Week 9 Lab: Ship Your Model

## Objective
Expose your model as a working API and a demo, and check its memory/behavior outside a notebook.

## Tasks

### Task 1: Gradio demo
Wrap your fine-tuned model (or base SmolLM2) in a `ChatInterface`. Launch with `share=True`. Visit the URL, run 3 exchanges, screenshot the result.

### Task 2: FastAPI endpoint
Write `app.py` with a `POST /generate` endpoint (prompt + max_new_tokens, capped at 512). Run it with uvicorn in the background. Test:
```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "The future of AI is", "max_new_tokens": 50}'
```
Show the response.

### Task 3: Error handling
- Send a request with an empty prompt. What happens?
- Send `max_new_tokens: 100000`. Does your cap save you?
- Add try/except so bad requests return a clean JSON error, not a crash. Re-test.

### Task 4: Memory check
```python
m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
print(f"{m.get_memory_footprint()/1e6:.0f} MB fp16")
q = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct",
     quantization_config=BitsAndBytesConfig(load_in_4bit=True))
print(f"{q.get_memory_footprint()/1e6:.0f} MB 4bit")
```
Record both numbers.

### Task 5: Load test (tiny)
Send 10 requests in a row (a loop). Measure total time. Report average seconds/request. Write one sentence on whether that latency would work for a real product.

## Deliverable
`app.py`, screenshots, and a notebook `week9_lab.ipynb` with memory numbers + load test.

## Check Yourself
- [ ] Does `curl` against your API work?
- [ ] Did you handle the bad requests without a crash?
- [ ] Do you know your model's memory footprint in fp16 vs 4-bit?
