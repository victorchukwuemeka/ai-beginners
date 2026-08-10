# Week 4: Tokenizers, Models, and Memory

## Core Concepts

### The Two Objects You Load
Last week you used `pipeline()` — one line that hides everything. This week you meet the two real objects underneath it: the **tokenizer** and the **model**. Understanding these two is what lets you build your own generation loop (Week 5) and fine-tune (Week 7).

### The Tokenizer: Text ↔ Numbers
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

# encode: text → token IDs (the numbers the model sees)
ids = tok.encode("Hello world")
print(ids)                                    # [15496, 995]

# decode: IDs → text (the numbers back to words)
print(tok.decode(ids))                        # "Hello world"

# see the raw tokens
print(tok.convert_ids_to_tokens(ids))         # ['Hello', 'Ġworld']
```
Key facts:
- Tokens are subword chunks, not words. `Ġ` is GPT-2's marker for a space.
- `tok.vocab_size` = how many distinct tokens the model can output (GPT-2: 50,257).
- Tokenizers get *configured* with padding and truncation for batches:
```python
enc = tok(["hi", "hello there"], padding=True, return_tensors="pt")
print(enc["input_ids"].shape)      # (2, 3) — batch 2, padded to length 3
print(enc["attention_mask"])       # 1s for real tokens, 0s for padding
```

### The Model: Token IDs → Scores
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
logits = model(**enc).logits
print(logits.shape)   # (2, 3, 50257) — batch, positions, vocab scores
```
For every position, the model outputs a score for every token in the vocabulary. `50257` scores = "how much the model wants each word next." Softmax turns scores into probabilities; that's what we sample from.

### `AutoModel` vs `AutoModelForCausalLM`
`transformers` uses `AutoModel*` to pick the right class from a model card:
- `AutoModelForCausalLM` — decoder-only generators (GPT-2, Llama, Qwen) → `generate()`
- `AutoModelForSequenceClassification` — classifiers (DistilBERT) → `...logits` for labels
- `AutoModel` — plain embeddings (Week 6 building block)
- `AutoModelForQuestionAnswering` — extractive Q&A
- `AutoModelForSeq2SeqLM` — encoder-decoder (T5, BART)

The architecture dictates which head you load. Using the wrong one is the #1 beginner error.

### Moving to GPU and Checking Memory
```python
model = model.to("cuda")      # GPU; CPU by default
print(model.get_memory_footprint() / 1e6, "MB")
```
Memory ≈ **2 bytes per parameter** at 16-bit (fp16):
- 135M → ~0.3 GB (laptop)
- 1.5B → ~3 GB (modest GPU)
- 8B → ~16 GB (serious GPU or quantized — Week 9)

Check `model.num_parameters()` to see for yourself.

### `generate()` — the token loop you'll build in Week 5
`pipeline()` and `model.generate()` both run the auto-regressive loop:
```python
out = model.generate(
    enc["input_ids"],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tok.eos_token_id,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Key Takeaway
`pipeline()` is sugar. Under it: a **tokenizer** (text ↔ IDs) and a **model** (IDs → vocab scores). Shapes to know by heart: input `(batch, seq)`, logits `(batch, seq, vocab)`, memory ≈ 2 bytes/param. If you can load these two objects and decode their outputs, you own the machinery.

---

## Common Pitfalls

### Pitfall 1: Wrong `AutoModel*` class
Loading a generator with `AutoModel` gives you no `generate()`. Match the head to the architecture.

### Pitfall 2: Forgetting `return_tensors="pt"`
The model needs tensors, not lists. Always pass `return_tensors="pt"` (or `"tf"`).

### Pitfall 3: No `pad_token` on GPT-2/SmolLM
They often lack one. Set `tok.pad_token = tok.eos_token` before batching.

### Pitfall 4: Logits ≠ probabilities
Logits are raw scores, can be any magnitude. Softmax them before interpreting.

### Pitfall 5: Forgetting the model stays on CPU
Default is CPU. `.to("cuda")` on a GPU machine — 10–50× faster.

---

## Interactive Exercises

1. Tokenize a sentence, print tokens, and reconstruct the text. Where did the spaces go?
2. Load a model, print `model.num_parameters()` and `get_memory_footprint()`. Confirm the 2 bytes/param rule.
3. Get logits for a 3-token prompt. Print the shape and read it aloud.
4. Load a *classifier* (`AutoModelForSequenceClassification`) and a *generator*, compare their outputs' shapes.
5. Set `tok.pad_token = tok.eos_token`, batch 3 prompts, check `input_ids.shape`.

---

## Quick Reference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
enc = tok(["a", "b c"], padding=True, return_tensors="pt").to("cuda")
logits = model(**enc).logits
out = model.generate(**enc, max_new_tokens=40, do_sample=True, temperature=0.8)
print([tok.decode(o, skip_special_tokens=True) for o in out])
```

---

## Further Reading
- HF course, tokenizer chapter: https://huggingface.co/learn/nlp-course
- AutoModel docs: https://huggingface.co/docs/transformers/model_doc/auto
