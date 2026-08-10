# Week 5: Inference Pipelines

## Core Concepts

### Beyond the Magic Function
`pipeline()` hides the real work. Week 5 opens the hood so you understand exactly what runs — and can customize it. The full inference path is:

```
text → [tokenizer] → token IDs → [model forward] → logits
     → [sampling] → next token → [repeat] → [stop] → [decode] → text
```

### Step 1: Tokenization
Text becomes integer IDs using a learned vocabulary.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
ids = tokenizer.encode("Hello world")
print(ids)                      # [15496, 995]
print(tokenizer.decode(ids))    # "Hello world"
print(tokenizer.convert_ids_to_tokens(ids))  # ['Hello', 'Ġworld']

# batching: pad to same length, mask the padding
enc = tokenizer(["hi", "hello there"], padding=True, return_tensors="pt")
print(enc["input_ids"].shape)   # (2, 3)
print(enc["attention_mask"])
```

### Step 2: The Forward Pass
Token IDs → one-hot → embedding lookup → transformer layers → **logits** (raw scores for every word in the vocabulary).

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
logits = model(**enc).logits
print(logits.shape)   # (2, 3, 50257)  — batch, positions, vocab size
```

For each position, `50257` scores = "how much the model wants each word next". This is the probability distribution we sample from.

### Step 3: Sampling Strategies
| Strategy | What It Does | Typical Use |
|----------|-------------|-------------|
| **Greedy** | Always pick the highest score | Deterministic, often repetitive |
| **Temperature** | Scale scores before softmax; high T = flatter = more random | Creative vs factual |
| **Top-k** | Only consider the top k tokens | Trims the tail of unlikely junk |
| **Top-p** (nucleus) | Consider the smallest set whose probabilities sum to p | Default in most models (0.9) |
| **Beam search** | Keep several candidate sequences | Best for translation/summarization |

```python
outputs = model.generate(
    enc["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 4: Auto-Regression (the loop)
A causal LM generates one token at a time: it feeds the previous output back in. That's why generation is sequential and why longer outputs cost more time. `generate()` runs this loop for you.

### Step 5: Decoding
Token IDs → text via the same tokenizer's vocabulary. (Note the `Ġ` — GPT-2's space token. That's the subword tokenization from Week 4, visible.)

### GPU vs CPU, Batching
- `model.to("cuda")` moves weights to GPU. GPU is ~10–50× faster for this math.
- **Batching**: process many sequences at once → same GPU, more throughput. That's why the input shape has a batch dimension.
- **Memory**: a 7B model at 16-bit ≈ 14 GB just to hold weights. This is why people quantize (Week 9).

---

## Key Takeaway
The magic of `pipeline()` is just: tokenize → forward → sample → decode, repeated. Knowing each stage lets you read errors, control generation, and talk about models precisely. The "inference pipeline" everyone talks about is literally these five steps wired together.

---

## Common Pitfalls

### Pitfall 1: Forgetting the batch dimension
The model expects `(batch, seq)`. Always `return_tensors="pt"` from the tokenizer.

### Pitfall 2: Padding side mismatches
Some models pad on the left, some right. `tokenizer.padding_side` matters for generation (left-pad for decoder-only).

### Pitfall 3: No `pad_token` on GPT-2
GPT-2 has no pad token. Set `pad_token_id=tokenizer.eos_token_id` (or add one).

### Pitfall 4: `model.generate` vs calling the model
`model(**enc)` returns logits for the *next* token at every position. `model.generate(...)` loops sampling until EOS or max tokens. Different tools, different jobs.

---

## Interactive Exercises

1. Tokenize a sentence, print tokens, and reconstruct the text. Where do spaces go?
2. Get logits for a short prompt. Find the top-5 next-token probabilities (use `torch.softmax`).
3. Generate with greedy vs temperature=0.8 vs top_p=0.9. Compare outputs.
4. Batch 3 prompts at once. Check the output shape is `(3, ...)`.
5. Break generation into a manual loop: `input_ids = model.generate(...)` one token at a time, appending each. (For eager learners.)

---

## Quick Reference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
m   = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
enc = tok(["a", "b c"], padding=True, return_tensors="pt").to("cuda")
out = m.generate(**enc, max_new_tokens=40, do_sample=True, temperature=0.8)
[tok.decode(o, skip_special_tokens=True) for o in out]
```

---

## Further Reading
- HF course, tokenizers chapter: https://huggingface.co/learn/nlp-course
- "How do transformers generate text?" (Hugging Face blog): https://huggingface.co/blog/how-to-generate
