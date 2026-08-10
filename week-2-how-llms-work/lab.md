# Week 2 Lab: Tokens, Embeddings, and Next-Token Prediction

## Objective
See the LLM machinery with your own eyes — tokens, embeddings, and next-token probabilities — before you run a full model next week.

## Setup
`pip install transformers torch` (or run in Colab). We'll use GPT-2 — small enough for any machine.

## Tasks

### Task 1: Watch tokens happen
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

text = "The cat sat on the mat"
ids = tok.encode(text)
print(ids)
print(tok.convert_ids_to_tokens(ids))
print(tok.decode(ids))
```
- Copy the token IDs and tokens into your notebook.
- Now tokenize "uncomfortable" and an emoji. What happens?
- Write one sentence about what you see (where do spaces/`Ġ` live?).

### Task 2: See the embeddings
```python
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("gpt2", output_hidden_states=True)
enc = tok(text, return_tensors="pt")
out = model(**enc)
emb = out.hidden_states[0]       # first layer embeddings
print(emb.shape)                 # (1, 6, 768) — batch, 6 tokens, 768-dim vector
print(torch.round(emb[0, 0, :5], decimals=2))
```
- The 6 tokens each became a 768-number vector. Print the shape aloud.

### Task 3: Predict the next token (the whole game)
```python
from transformers import AutoModelForCausalLM
lm = AutoModelForCausalLM.from_pretrained("gpt2")
enc = tok("The capital of France is", return_tensors="pt")
logits = lm(**enc).logits[0, -1, :]            # scores for the next token
probs = torch.softmax(logits, dim=-1)
top5 = torch.topk(probs, 5)
for p, i in zip(top5.values, top5.indices):
    print(f"{tok.decode(i):>12}  {p.item():.4f}")
```
- Write down the top 5 next tokens and their probabilities.
- Change the prompt to "The capital of Japan is". What changes?

### Task 4: Two important shapes
Print:
- `tok.vocab_size` — how many tokens can the model choose from?
- `emb.shape[-1]` — how big is one embedding?

## Deliverable
A notebook `week2_lab.ipynb` with the token outputs, embedding shape, and your top-5 next-token tables for both prompts.

## Check Yourself
- [ ] Did you see the tokenizer split words (including `Ġ`)?
- [ ] Do you know the embedding dimension of GPT-2 (768)?
- [ ] Could you explain to someone why the next token for "capital of France" was "is"/"Paris"?
