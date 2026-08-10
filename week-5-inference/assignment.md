# Week 5 Assignment: Your Own Generation Function

## Due
End of Week 5

## Objective
Write a function that does text generation start-to-finish — proof you understand the inference pipeline.

## Requirements

### 1. Write `generate(prompt, temperature=0.8, top_p=0.9, max_new_tokens=50)`
Build it with the tokenizer + model (no `pipeline()`), implementing:
- tokenize the prompt
- auto-regressive loop (one token at a time)
- temperature sampling using `torch.multinomial`
- top-p (nucleus) filtering: sort probs, keep the smallest set with cumulative prob ≥ p, renormalize
- stop at EOS or max tokens
- decode the final sequence

Include a `main` that calls it with a seed phrase.

### 2. Demo and compare
Run your function with:
- `temperature=0.0` (argmax, no sampling)
- `temperature=0.8`
- `temperature=1.5`
- `top_p=0.5` vs `top_p=0.95`

Copy all outputs. Write a short table: `setting | output | effect you observe`.

### 3. Stress test
What does your function do with an empty/whitespace prompt? With a prompt that hits the model's max context length? Document the errors and fix what's reasonable.

### 4. Reflection (150–300 words)
- What did building generation from scratch teach you that `pipeline()` wouldn't?
- Where in your loop would a real product save time (caching, batching, KV cache — name the concepts you noticed)?

## Grading
| Criterion | Weight |
|-----------|--------|
| Manual generation works (greedy + temperature + top-p) | 40% |
| Comparison table is evidence-based | 20% |
| Edge cases handled/documented | 15% |
| Code is clean, readable, runs | 15% |
| Reflection is specific | 10% |
