# Week 5 Lab: Build an Inference Pipeline Without `pipeline()`

## Objective
Recreate the inference path manually — tokenizer → forward pass → sampling → decode — so the machinery stops being magic.

## Tasks

### Task 1: Tokenizer round trip
```python
tok = AutoTokenizer.from_pretrained("gpt2")
tokens = tok.encode("Machine learning is the best")
print(tokens)
print(tok.convert_ids_to_tokens(tokens))
print(tok.decode(tokens))
```
Notice where spaces live. Write one sentence about what `Ġ` is.

### Task 2: Read the logits
```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
ids = tok("The next word is", return_tensors="pt")
logits = model(**ids).logits          # (1, 4, 50257)
next_logits = logits[0, -1, :]        # last position
import torch
probs = torch.softmax(next_logits, dim=-1)
top5 = torch.topk(probs, 5)
for p, i in zip(top5.values, top5.indices):
    print(f"{tok.decode(i):>10}  {p:.4f}")
```
Write down the top 5 next words and their probabilities.

### Task 3: Manual generation loop (one token at a time)
```python
prompt = "In the beginning there was"
input_ids = tok(prompt, return_tensors="pt").input_ids
for _ in range(20):
    logits = model(input_ids).logits[0, -1, :]      # next-token scores
    next_id = torch.argmax(logits).unsqueeze(0)     # greedy pick
    input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
print(tok.decode(input_ids[0]))
```
That loop is inference. Now modify it to add **temperature**: sample from `softmax(logits / T)` with `torch.multinomial`. Compare greedy vs T=0.8.

### Task 4: `generate()` does the same
```python
out = model.generate(tok(prompt, return_tensors="pt").input_ids, max_new_tokens=20)
print(tok.decode(out[0]))
```
Confirm it matches your greedy loop.

### Task 5: Batching
Generate for 3 different prompts in one call (pad inputs). Print all three outputs.

## Deliverable
A notebook `week5_lab.ipynb` with the manual loop working — this is your first from-scratch inference engine.

## Check Yourself
- [ ] Did your manual greedy loop match `model.generate` (Task 4)?
- [ ] Could you explain where temperature gets applied?
- [ ] Did you confirm the logits shape `(batch, seq, vocab)` by reading it aloud?
