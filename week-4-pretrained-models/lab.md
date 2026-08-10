# Week 4 Lab: Meet the Tokenizer and the Model

## Objective
Work with the two real objects under `pipeline()` — the tokenizer and the model — and measure their memory.

## Setup
`pip install transformers torch`. GPT-2 runs fine on CPU.

## Tasks

### Task 1: Tokenizer round trip
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

tokens = tok.encode("Machine learning is the best")
print(tokens)
print(tok.convert_ids_to_tokens(tokens))
print(tok.decode(tokens))
```
- Copy the tokens list.
- Write one sentence about what `Ġ` is.
- Now encode: `"uncomfortable"`, an emoji, and a URL. Compare token counts.

### Task 2: Shape literacy (the two shapes that matter)
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
enc = tok(["hi", "hello there"], padding=True, return_tensors="pt")
logits = model(**enc).logits

print("input ids:", enc["input_ids"].shape)   # (2, 3)
print("logits:   ", logits.shape)             # (2, 3, 50257)
print("vocab size:", tok.vocab_size)
```
Write the two shapes down and label each dimension out loud: (batch, seq) and (batch, seq, vocab).

### Task 3: Softmax makes scores → probabilities
```python
probs = torch.softmax(logits[0, -1, :], dim=-1)   # last position, first sequence
top5 = torch.topk(probs, 5)
for p, i in zip(top5.values, top5.indices):
    print(f"{tok.decode(i):>12}  {p.item():.4f}")
```
These are the model's top 5 candidate next tokens. Note: we did NOT call `generate()` — we read the raw distribution.

### Task 4: Memory check
```python
print("parameters:", model.num_parameters())
print("memory MB: ", model.get_memory_footprint() / 1e6)
```
- Check the rule: params × 2 bytes ≈ memory footprint. Does it hold?
- Do the same for `HuggingFaceTB/SmolLM2-135M` and compare the two numbers.

### Task 5: Generate with the raw objects
```python
tok.pad_token = tok.eos_token
enc = tok("The future of AI is", return_tensors="pt")
out = model.generate(enc.input_ids, max_new_tokens=30, do_sample=True, temperature=0.8,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```
Same loop `pipeline()` runs — but now you're calling the parts yourself.

## Deliverable
A notebook `week4_lab.ipynb` with all five tasks, shapes labeled, and both models' memory numbers.

## Check Yourself
- [ ] Can you say the two shapes out loud with their meanings?
- [ ] Did your memory check confirm ~2 bytes/parameter?
- [ ] Did your manual generate work without `pipeline()`?
