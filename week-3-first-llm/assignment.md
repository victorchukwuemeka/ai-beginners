# Week 3 Assignment: Prove You Can Run a Model

## Due
End of Week 3

## Objective
Demonstrate you can load, run, and *compare* pretrained models — the core skill of inference.

## Requirements

### 1. Model zoo (run each, one output each)
- `text-generation` with GPT-2 or SmolLM2
- `sentiment-analysis` on 5 of your own sentences (mix positive/negative/neutral)
- `question-answering` over a context you wrote
- `zero-shot-classification` with labels of your choosing

### 2. Generation settings experiment
Take one seed phrase. Run it under these settings and copy the outputs:
- `temperature=0.0` (greedy)
- `temperature=0.8, do_sample=True`
- `temperature=1.5, do_sample=True`
- `max_new_tokens=10` vs `max_new_tokens=150`

Write a paragraph explaining what each knob did, citing your actual outputs.

### 3. Model comparison
Run the same seed through two different small models (e.g., GPT-2 and SmolLM2-135M-Instruct). Compare in a small table: coherence, repetitiveness, "weirdness" (your subjective 1–5). Write 2–3 sentences of analysis.

### 4. Reflection (150–300 words)
- What's now obvious that was mysterious before?
- What do you still not understand (that later weeks will answer)?

## Grading
| Criterion | Weight |
|-----------|--------|
| All four pipelines run correctly | 30% |
| Settings experiment shows real differences with evidence | 25% |
| Model comparison is thoughtful | 20% |
| Code is clean and runs | 15% |
| Reflection is specific | 10% |
