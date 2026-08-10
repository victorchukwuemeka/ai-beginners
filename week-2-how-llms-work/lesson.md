# Week 2: How LLMs Work

## Core Concepts

### The One Idea: Next-Token Prediction
An LLM does exactly one thing: given some text, predict the most likely **next token** (a small piece of a word). Everything else — chat, coding, reasoning — is that simple loop running over and over.

```
"The cat sat on the ___"  →  model  →  "mat" (90%), "floor" (7%), "hat" (2%)...
```

It feels like intelligence because the loop runs a million times and the model has learned patterns from billions of sentences.

### The Pipeline of an LLM
```
text → [tokens] → [embeddings] → [transformer layers / attention] → [next-token scores] → [sampling] → text
```

Four pieces to understand:

**1. Tokens**
The model doesn't read letters or words — it reads *tokens*, small subword chunks from a learned vocabulary.

| Text | Tokens |
|------|--------|
| "Hello world" | `hello`, `Ġworld` |
| "uncomfortable" | `un`, `comfort`, `able` |

The tokenizer (a lookup table) converts text → token IDs and back. Rule of thumb: ~4 characters or ~0.75 words per token.

**2. Embeddings**
Each token ID is mapped to a **vector of numbers** (e.g., 768 numbers) that encodes its meaning. Similar-meaning tokens get similar vectors — that's how the model captures *semantics*, not just spelling. This is Week 5's building block for search.

**3. Transformer layers (attention)**
The embedding flows through many layers. **Attention** is the trick that lets each token "look at" all the other tokens in the sentence and decide which matter most — so "bank" next to "river" means the riverside, while "bank" next to "money" means the institution. Context is the whole game.

**4. Scores and sampling**
The last layer outputs a score for every token in the vocabulary (e.g., 50,000 numbers). Softmax turns them into probabilities. We then pick the next token:
- **Greedy** — always the highest probability
- **Sampling** — random, weighted by the probabilities (temperature/`top_p` control how "random")

### Training vs Inference (the two phases)
| Phase | What Happens | Cost |
|-------|-------------|------|
| **Training** | Model adjusts millions of weights to reduce error over huge data | Expensive, offline |
| **Inference** | Model uses fixed weights to predict the next token on your input | Cheap, online |

An LLM like GPT was **trained** once (billions of dollars of GPUs). You **use** it via inference. Later weeks: RAG (Week 6) and fine-tuning (Week 7) both build on this split.

### Sizes
A "7B model" has 7 billion numbers (weights). Bigger = more knowledge, but needs more memory (~2 bytes per parameter at 16-bit):
- 135M model ≈ 0.3 GB — runs on a laptop, your course workhorse
- 1.5B model ≈ 3 GB — needs a modest GPU
- 8B model ≈ 16 GB — needs a serious GPU or quantization

---

## Key Takeaway
An LLM predicts the next token, repeatedly. Text → tokens → embeddings → attention → scores → sample. Training learned the weights; inference uses them. If you can explain those four stages, you understand how every LLM works.

---

## Common Pitfalls

### Pitfall 1: Thinking it "understands" like a person
It predicts patterns. Useful, but not truth — that's why outputs need checking (Week 8).

### Pitfall 2: Confusing training and inference
"Can I fine-tune by just asking it to answer well?" No — that's prompting (inference). Fine-tuning is *training* (Week 7).

### Pitfall 3: Tokens ≠ words
Emoji, URLs, and rare words take multiple tokens and eat context. Why it matters: cost and context limits.

### Pitfall 4: Assuming bigger is always better
Bigger models are slower and pricier. A 135M model does a surprising amount with RAG on top (Week 6).

---

## Interactive Exercises

1. Open a tokenizer (e.g., https://platform.openai.com/tokenizer or the `tiktoken` library) and count tokens for: a common sentence, an emoji, a URL. Note the differences.
2. Copy a 100-word paragraph into any LLM and ask it to predict the "next word" after every sentence break. Does it fit the context each time?
3. Ask the same question twice at different temperature settings (0 and 0.9). Compare.
4. In your own words, explain to a friend: why does "bank" mean different things next to "river" vs "money"? (That's attention.)

---

## Quick Reference

```
tokenizer: text  → token IDs          | tokenizer: token IDs → text
embedding: token → vector             | attention: token sees its context
sampling:  scores → next token        | loop: repeat until stop
```

---

## Further Reading
- 3Blue1Brown, "How LLMs work" (visual, ~30 min): https://www.youtube.com/watch?v=LPZh9BOjkQs
- Hugging Face course, transformer chapter: https://huggingface.co/learn/nlp-course
