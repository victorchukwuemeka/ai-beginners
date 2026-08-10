# Week 8 Lab: Score Your Model, Not Your Feelings

## Objective
Build a real eval harness for your Week 7 fine-tuned model and decide — with numbers — whether LoRA worked.

## Tasks

### Task 1: Build a held-out test set
Take your Week 7 dataset. Split it **before training matters**: 150+ train, 20 held-out test (do NOT fine-tune on these). If you already fine-tuned on everything, write 20 fresh prompts now that follow the same behavior rule — and be honest that they're fresh, not held-out.

### Task 2: Binary scoring
Write a scorer that checks your target behavior on each of the 20 test prompts:
```python
def follows_behavior(output: str) -> bool:
    # e.g., output.count("•") == 3 or "ends with a question mark" or parses as JSON
    ...
```
Score base model and fine-tuned model on all 20. Produce:
```python
accuracy_base   = sum(follows_behavior(o) for o in base_outputs)   / 20
accuracy_tuned  = sum(follows_behavior(o) for o in tuned_outputs)  / 20
```

### Task 3: Add a content score
Not enough to match the format — is the content right? For 10 of the 20 prompts where you know the answer, score `content_correct` yes/no for both models. Report format-accuracy and content-accuracy separately.

### Task 4: LLM-as-judge
For 5 open-ended prompts, have a second model (or a rubric you apply) score both models' outputs 1–5. Show the scores side by side. Do you agree with the judge? Where would you override it?

### Task 5: Write the verdict
One paragraph: **did fine-tuning work, by how much, and measured on what?** Give the exact numbers.

## Deliverable
A notebook `week8_lab.ipynb` with: test set, scorer, both models' scores, and a written verdict with numbers.

## Check Yourself
- [ ] Did you test on prompts the tuned model never saw?
- [ ] Do you have a number, not a vibe?
- [ ] Could someone re-run your notebook and get the same method?
