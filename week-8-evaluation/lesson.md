# Week 8: Evaluation

## Core Concepts

### Why "It Looks Good" Is Not a Metric
You cannot tell if a model improved by eyeballing 3 outputs. You need a **test set** (examples the model never saw during training), a **metric**, and a **baseline to beat**. This is how you know your Week 7 fine-tune actually worked.

### The Three Building Blocks

**1. Held-out test set** — data not used in training. Split: train / validation / test (e.g., 70/15/15). Test set is sacred: only touch it at the end.

**2. Baseline** — a reference to beat. "Fine-tuned vs base model" or "our RAG vs no-RAG". A number is meaningless without a baseline.

**3. Metric** — a number that maps outputs to quality. Pick by task type.

### Metrics by Task Type

| Task | Metric | What It Measures |
|------|--------|-----------------|
| Classification | **Accuracy, F1, precision, recall** | Did it pick the right label? (F1 handles unbalanced classes) |
| Text generation | **BLEU / ROUGE** | How much does the output overlap the reference text? (rough, but cheap) |
| Language modeling | **Perplexity** | How surprised is the model by real text? Lower = better |
| Retrieval | **Recall@k** | Did the right document get retrieved in the top-k? |
| Anything open-ended | **LLM-as-judge / rubric** | A strong model (or human) scores on criteria |

### Classification Metrics in One Minute
| Metric | Definition |
|--------|-----------|
| Precision | Of everything labeled positive, how much was actually positive? |
| Recall | Of everything actually positive, how much was found? |
| F1 | Harmonic mean of precision & recall — one number to balance both |

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]
print(accuracy_score(y_true, y_pred))    # 0.6
print(precision_score(y_true, y_pred))   # 0.667
print(recall_score(y_true, y_pred))      # 0.667
print(f1_score(y_true, y_pred))          # 0.667
```

### Regression Testing (the discipline)
Whenever you improve the model (new fine-tune, new prompt, better RAG), **re-run the same test set** and confirm nothing got worse. This is how teams ship without breaking things — your RAG/system quality is only as good as your eval suite.

### Eval-Driven Development
```
1. Write the test set first  ← do this before improving anything
2. Measure baseline
3. Make one change (new chunking, LoRA, prompt)
4. Re-measure on the SAME test set
5. Keep the change only if it helps
```

### LLM-as-Judge (when you have no labels)
For open-ended output, write a rubric and have a strong model score each response:

```python
rubric = """Score 1-5 on: (1) follows format (2) correct content (3) tone. 
Output: {output}  Answer: {expected}"""
```

Caveats: judges have biases (favor longer answers, favor the model they're derived from). Use multiple judges or spot-check with humans.

---

## Key Takeaway
Evaluation = **test set + baseline + metric**, run before and after every change. "Seems better" is not evidence; a metric on unseen data is. Regression testing makes improvements safe to keep.

---

## Common Pitfalls

### Pitfall 1: Testing on training data
A model scores 100% on its own training data. Only held-out examples count.

### Pitfall 2: Comparing different test sets
You changed the test set between runs → the numbers aren't comparable. Freeze one test set.

### Pitfall 3: Optimizing the metric, not the product
If your rubric rewards length, models get verbose. Evaluate what actually matters to users.

### Pitfall 4: No baseline
"Accuracy 0.82" alone is meaningless. "0.82 vs base model 0.61" is meaningful.

### Pitfall 5: Changing multiple things at once
Change chunking AND model AND prompt, get a better score → you don't know what helped. One change per experiment.

---

## Interactive Exercises

1. Build a 20-example test set for your Week 7 fine-tune. Score base vs tuned. Any surprises?
2. Compute accuracy, precision, recall, F1 on a made-up 10-item classification result. Verify with sklearn.
3. Take one of your RAG answers. Give it to a second model with a rubric and get a score. Cross-check it yourself.
4. Freeze a test set and make one improvement to your RAG; measure before/after on the same 20 questions.

---

## Quick Reference

```python
# the eternal eval loop
test_set = [...]                      # 1. fixed, unseen
baseline = evaluate(base_model)       # 2. measure
change_model()                        # 3. one change
score = evaluate(changed_model)       # 4. re-measure
assert score > baseline               # 5. keep only if better
```

---

## Further Reading
- MLU-Explain: https://mlu-explain.github.io/precision-recall/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate
