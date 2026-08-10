# Week 6: RAG — Retrieval-Augmented Generation

## Core Concepts

### The Problem
An LLM knows what it learned during training — which is static and may be wrong or outdated. You can't (and shouldn't) stuff all your documents into the prompt every time. **RAG** answers: how do we give the model *your* data, on demand?

### The One-Sentence Definition
RAG = retrieve the most relevant pieces of your documents, paste them into the prompt as context, then let the model answer using them.

```
user question
     │
     ▼
[embed the question] ────────────────┐
                                    │
[embed your documents] ──► vector DB ──► top-k similar chunks
                                    │
                                    ▼
[prompt: question + retrieved chunks] ──► model ──► grounded answer
```

### Step 1: Embeddings (turning text into numbers)
An embedding is a vector (e.g., 384 or 768 numbers) that captures meaning. Similar texts get similar vectors. From Week 2: we measure similarity with the **dot product / cosine similarity**.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")   # tiny, 384-dim
v1 = model.encode("I love pizza")
v2 = model.encode("Pizza is my favorite food")
v3 = model.encode("The stock market is volatile")
import numpy as np
print(v1 @ v2)   # high — similar meaning
print(v1 @ v3)   # low — unrelated
```

### Step 2: Chunking
Documents are too long to embed or retrieve whole. Split them into chunks (~200–500 words) with some overlap, so a question about any part can still match.

### Step 3: The Vector Database (FAISS)
FAISS (Facebook AI Similarity Search) stores embeddings and returns the top-k closest to a query, fast.

```python
import faiss
import numpy as np

chunks = ["Paris is the capital of France.", "The Eiffel Tower is in Paris.", ...]
embeds = np.array([model.encode(c) for c in chunks]).astype("float32")
index = faiss.IndexFlatL2(embeds.shape[1])
index.add(embeds)

q_emb = model.encode("Where is the Eiffel Tower?").astype("float32")
dists, idxs = index.search(q_emb.reshape(1, -1), k=2)   # top-2
for i in idxs[0]:
    print(chunks[i])
```

### Step 4: The Prompt Assembly
```python
context = "\n\n".join(chunks[i] for i in idxs[0])
prompt = f"""Answer using ONLY the context below. Cite where you got it.
Context:
{context}

Question: {q}
Answer:"""
```
Now any chat model can answer with your data as ground truth.

### Step 5: Grounding and Citation
Because we put retrieved text in the prompt, the model can say *where* it got each fact. This is the difference between a model guessing and a model **grounded in your documents** — the heart of enterprise AI.

### The Golden Rule of Techniques
```
Use prompting first. Add RAG second. Fine-tune last.
```
RAG wins when your data *changes* (news, tickets, internal docs). Fine-tuning (Week 7) wins for *form* — tone, structure, style. Retrieval is for facts.

---

## Key Takeaway
RAG is: embed documents → index them → retrieve the best chunks for a question → stuff them into the prompt. It's maybe 40 lines of code and it turns a generic model into an expert on *your* data, with citations, and no training required.

---

## Common Pitfalls

### Pitfall 1: Bad chunking
Too big → irrelevant context floods the prompt. Too small → missing information. Start at ~300 words with 50-word overlap.

### Pitfall 2: Ignoring retrieval quality
If the wrong chunks are retrieved, the answer is wrong no matter how good the model is. **Debug retrieval first**, generation second.

### Pitfall 3: Cramming everything in one prompt
Context windows are finite. Retrieve top-3 to top-5 chunks, not 50.

### Pitfall 4: Not evaluating
"Is the answer right?" requires a test set with known answers. We do that properly in Week 8 — but keep a list of 10 questions from day one.

---

## Interactive Exercises

1. Build the 40-line RAG loop above on 5 paragraphs of text you write about a hobby.
2. Ask a question with a direct answer in the text; then one where the answer spans two chunks. Compare.
3. Print which chunks were retrieved. Are they the right ones?
4. Ask a question with no answer in your documents. What does the model do?

---

## Quick Reference

```bash
pip install sentence-transformers faiss-cpu
```

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
emb = SentenceTransformer("all-MiniLM-L6-v2")
idx = faiss.IndexFlatL2(384)
# ... embed chunks → idx.add(...) → idx.search(q, k) → prompt → model
```

---

## Further Reading
- Hugging Face RAG tutorial: https://huggingface.co/learn/rag
- FAISS docs: https://github.com/facebookresearch/faiss
