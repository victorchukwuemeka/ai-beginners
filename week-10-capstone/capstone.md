# Capstone: Build One Real ML/AI System

## Due
End of Week 10 (final submission)

## Objective
Build **one complete, working AI system** that solves a real problem you care about — and document it so someone else could use and evaluate it.

## Choose Your Project
Pick ONE that stretches you but is achievable with free hardware:

### Option A: Domain RAG Assistant
- Pick a real document set you have access to (course notes, manuals, papers, internal docs)
- Build: chunking → embeddings → FAISS → retrieval → generation with citations
- Extras that push it further: hybrid search, reranking, eval suite

### Option B: Fine-Tuned Specialist
- Pick a behavior the base model won't reliably do (structured output, persona, domain style)
- Build: dataset (150+) → LoRA fine-tune → held-out evaluation → deployed demo
- Extras: iterate on dataset based on eval, compare two LoRA ranks

### Option C: Both (RAG + fine-tuning)
- Fine-tune for form, add RAG for facts
- Evaluate the combined system vs either alone

## Required Deliverables

### 1. Repository (GitHub/GitLab, or zipped with `.git`)
```
capstone/
├── README.md          # what it is, how to run, architecture diagram
├── data/              # your data (or a script that fetches it)
├── code/              # the system (runs end-to-end)
├── eval/              # test set + evaluation report
├── serving/           # the API/demo
└── reflection.md      # your write-up
```
Version control throughout: descriptive commits, at least one experiment on a branch merged or rejected.

### 2. Architecture (in README)
A diagram or clear text of the pipeline:
`input → [component] → [component] → output`, naming each stage (chunker, embedder, index, retriever, generator).

### 3. Evaluation
- A frozen test set (10+ examples with known expected behavior)
- A baseline measured before your main improvement
- After measurement on the SAME test set
- Honest failure analysis (2+ failure patterns with examples)
- An `EVALUATION.md` with the numbers

### 4. Working demo
A Gradio demo or FastAPI endpoint that actually runs. A screenshot or public URL.

### 5. Reflection (400+ words)
- What problem did you solve, and for whom?
- What was the hardest technical step and how did you get past it?
- What did evaluation teach you that eyeballing didn't?
- What would you do differently with 4 more weeks?
- What's the single number that proves your system is better than baseline?

## Grading Criteria

| Criterion | Weight |
|-----------|--------|
| System works end-to-end (runnable) | 25% |
| Architecture clear and documented | 15% |
| Evaluation: frozen test set, baseline, before/after, honest failures | 30% |
| Repository discipline: commits, branches, README | 15% |
| Reflection is specific and honest | 15% |

## Milestones (don't cram)
- Week 8: pick the project, write the README skeleton + test set
- Week 9: main pipeline working
- Week 10: evaluation + demo + reflection + polish

## Final Reminder
> The point is not a working demo — it's a **documented, evaluated system** whose claims you can defend with numbers. A model that runs is a notebook; a model you can prove is better is a portfolio piece.
