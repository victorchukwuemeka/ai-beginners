# Week 6 Lab: Q&A Bot Over Your Own Documents

## Objective
Build a working RAG pipeline — embeddings, FAISS, retrieval, prompt assembly — that answers questions over your own text.

## Tasks

### Task 1: Your corpus
Write 8–12 short paragraphs about a topic you know well (or paste articles you have). Save as a list of paragraphs. (Each paragraph = one chunk, ~150–300 words.)

### Task 2: Embed and index
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

emb = SentenceTransformer("all-MiniLM-L6-v2")
embeds = np.array([emb.encode(c) for c in chunks]).astype("float32")
index = faiss.IndexFlatL2(embeds.shape[1])
index.add(embeds)
print("indexed", len(chunks), "chunks")
```

### Task 3: Retrieval sanity check
```python
q = "..."                       # a question that should match chunk #2
dists, idxs = index.search(emb.encode(q).reshape(1, -1).astype("float32"), k=3)
for i in idxs[0]:
    print("—", chunks[i][:80])
```
Is the right chunk retrieved? Try 3 different questions. Record what's retrieved for each.

### Task 4: Assemble and answer
Wire retrieval into a generation call (use any chat model you have — SmolLM2, or an API):
```python
context = "\n\n".join(chunks[i] for i in idxs[0])
prompt = f"Answer using only the context.\n\nContext:\n{context}\n\nQ: {q}\nA:"
# → feed to your generation model
```
Answer 5 of your questions. Copy question → retrieved chunks → answer.

### Task 5: Failure case
Ask a question NOT in your documents. What does the answer look like? Write one sentence on why.

## Deliverable
A notebook `week6_lab.ipynb` that runs end-to-end: chunks → index → retrieve → answer.

## Check Yourself
- [ ] Did you print the retrieved chunks and confirm they're relevant?
- [ ] Could a stranger run your notebook and ask their own question?
- [ ] Do you understand why bad retrieval → bad answers?
