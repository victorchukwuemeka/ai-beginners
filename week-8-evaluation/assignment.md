# Week 8 Assignment: An Evaluation Report

## Due
End of Week 8

## Objective
Produce a documented evaluation of ONE system you've built (your RAG, your fine-tuned model, or both) — the way a real team would.

## Requirements

### 1. Evaluation plan (write before running)
- What are you evaluating? (system + version)
- What's the baseline to beat?
- Test set: how many examples, where from, are they unseen?
- Metrics: which ones and why (matching the task type)
- Rubric/LLM-as-judge criteria if used

### 2. Run the eval
Run your plan. Capture raw outputs in the notebook.

### 3. The report (`EVALUATION.md`)
- Summary table: `system | metric | score | vs baseline`
- Confusion-style breakdown for classification (or format/content split for generation)
- 3–5 example outputs with your scoring and a note on why you scored them that way
- Failure patterns: what kinds of examples fail? (name at least 2 patterns)
- Honest limitations: what does your metric NOT measure?

### 4. One improvement round (regression testing)
Make exactly one improvement (better chunking, different LoRA rank, better prompt). Re-run the SAME test set. Report before/after. Keep it only if it helped — and say so either way.

### 5. Reflection (150–300 words)
- What did the metric catch that your eyes missed?
- What would you measure differently next time?

## Grading
| Criterion | Weight |
|-----------|--------|
| Plan written before running (frozen test set, baseline, metrics) | 25% |
| Evaluation executed correctly with raw outputs | 25% |
| Report is honest, specific, includes failure patterns | 25% |
| Improvement round with before/after on same test set | 15% |
| Reflection is specific | 10% |
