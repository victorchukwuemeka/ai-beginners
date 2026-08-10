# Week 6 Assignment: A Document Q&A System

## Due
End of Week 6

## Objective
Ship a small but complete RAG system over your own documents and prove retrieval quality.

## Requirements

### 1. The system
Build a `rag.py` (or notebook) that:
- Loads a document set (use at least 10 chunks of real content)
- Chunks with overlap
- Embeds with `all-MiniLM-L6-v2`
- Indexes with FAISS
- Answers questions by retrieval + generation, **citing the source chunk** in the output (e.g., `[Source: chunk 3]`)

### 2. A test set of 10 questions
Write 10 questions with known answers that live in your documents:
- 4 directly-answerable (one chunk)
- 3 spanning two chunks
- 2 edge cases (nearly-identical content, tricky wording)
- 1 out-of-scope (answer NOT in your docs)

### 3. Retrieval report
For each question, record which chunks were retrieved and whether the correct one was in the top-3. Report: **recall@3** = (# questions where the right chunk was retrieved) / 10.

### 4. Failure analysis
For the 2–3 questions with worst retrieval or answers:
- Show the wrong chunk retrieved
- Explain why (chunking? wording? embedding limitation?)
- Make one improvement (better chunking, more overlap, top-k change) and re-test

### 5. Reflection (150–300 words)
- What was the weakest part of your RAG system?
- When would you prefer this over fine-tuning, and vice versa?

## Grading
| Criterion | Weight |
|-----------|--------|
| RAG system works end-to-end with citations | 30% |
| Test set is well-designed (10 Q, known answers) | 20% |
| Retrieval report with recall@3 | 20% |
| Failure analysis + improvement | 20% |
| Reflection is specific | 10% |
