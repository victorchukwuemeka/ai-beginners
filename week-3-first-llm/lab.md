# Week 3 Lab: Run Real Models

## Objective
Run four real pretrained models and see inference in action — your first step from "user of a chatbot" to "person who runs models."

## Setup
Colab or local. `pip install transformers torch`. The small models in this lab run fine on CPU; use `device=0` on GPU if available.

## Tasks

### Task 1: The 3-line generator
```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
out = gen("I asked my computer for advice and it said", max_new_tokens=50, do_sample=True)
print(out[0]["generated_text"])
```
Run it 3 times with `temperature=0.2`, `0.8`, `1.5`. Copy all three outputs. Write one sentence on how temperature changed the text.

### Task 2: Classifier
```python
cls = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
for s in ["This course is brilliant", "This is terrible", "It's okay I guess"]:
    print(s, "→", cls(s))
```
How confident is it on the neutral-ish sentence? (Look at the score.)

### Task 3: Extractive Q&A
```python
qa = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
ctx = "The Nile is the longest river in Africa, flowing north through eleven countries. Its primary sources are the White Nile and the Blue Nile."
print(qa(question="What is the longest river in Africa?", context=ctx))
print(qa(question="What flows north?", context=ctx))
print(qa(question="What is the tallest mountain?", context=ctx))   # NOT in the text
```
What does it do on the question with no answer? Write it down.

### Task 4: Zero-shot classification
```python
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
for s in ["Stocks rallied on strong earnings", "A new recipe for pasta", "The team scored in overtime"]:
    print(s, "→", zsc(s, candidate_labels=["business", "cooking", "sports"])["labels"][:2])
```
Notice: no training needed to use new labels — that's the "zero-shot" trick.

### Task 5: Model sizes
Load `HuggingFaceTB/SmolLM2-135M-Instruct` and generate text. Check the downloaded folder size (`!du -sh ~/.cache/huggingface` in Colab). Compare mentally to the ~16 GB a full Llama-8B would take.

## Deliverable
A notebook `week3_lab.ipynb` with all five tasks, including the temperature comparison and the no-answer Q&A result.

## Check Yourself
- [ ] Did you note how temperature changed the output?
- [ ] Did you see the model hallucinate/guess on the no-answer question?
- [ ] Could you now define "inference" from having done it?
