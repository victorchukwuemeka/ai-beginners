# Week 2: How LLMs Work

## Core Concepts

### 1. The One Idea: Next-Token Prediction

An LLM does exactly one thing: given some text, predict the most likely **next token** (a small piece of a word). Everything else — chat, coding, reasoning — is that simple loop running over and over.

```
"The cat sat on the ___"  →  model  →  "mat" (90%), "floor" (7%), "hat" (2%)...
```

Here's the full loop:

```
"The cat sat on the mat"     ← generated "mat", append it
"The cat sat on the mat and" ← generated "and", append it
"The cat sat on the mat and" → generate "snored"... keep going until a stop token
```

That's it. Generation is just: **predict the next token → append → predict again**. An LLM never "plans the whole answer" — it produces one token at a time, and each new token's prediction is conditioned on everything generated so far (plus your prompt). It feels like intelligence because the loop runs a thousand times and the model has learned patterns from billions of sentences.

---

### 2. Tokens: The Alphabet an LLM Reads

The model doesn't read letters or words — it reads **tokens**, small subword chunks from a learned vocabulary (e.g., ~50,000 entries for many models).

| Text | Tokens |
|------|--------|
| "Hello world" | `hello`, `Ġworld` |
| "uncomfortable" | `un`, `comfort`, `able` |
| "tokenization" | `token`, `ization` |

**Subword tokenization** (the actual algorithm, e.g. BPE) finds the vocabulary that compresses your training corpus best:
- **Common words** stay whole → efficient (1 token for "the").
- **Rare words** get split into meaningful pieces → "uncomfortable" → `un` + `comfort` + `able`.
- **Anything unseen** can always be assembled from pieces → no unknown words, ever.

This is why vocabularies are learned from data rather than hand-picked.

**The tokenizer** is the lookup table that converts text → token IDs and back. It's a separate component from the model — you load it alongside (you'll do exactly this in Week 3):

```text
"Hello world" → [15496, 995] → model → [123, 456, ...] → "world"
```

**Rule of thumb: ~4 characters or ~0.75 words per token.**

**Why tokens matter beyond curiosity:**
- **Cost & speed** — every token is billed and processed; fewer tokens = cheaper and faster.
- **Context limits** — a "2,000 token window" fits ~1,500 words, but an emoji or a URL can eat 3–6 tokens.
- **Batching math** — batches are padded to the longest sequence in the group; that's memory being wasted.

**Special tokens** are reserved IDs with jobs — `[CLS]`, `<s>`, `</s>`, `<pad>`, and importantly `<|endoftext|>`/`<|im_end|>`-style **stop tokens**. Generation stops when the model emits one. You'll encounter these the moment you inspect tokenizer outputs in the lab.

---

### 3. Embeddings: Giving Tokens Meaning

Each token ID is mapped to a **vector of numbers** — e.g., 768 numbers for a "small" model, 4096 for a big one. This vector encodes meaning: **similar-meaning tokens land close together in vector space**.

```text
"king"  → [0.32, -0.11, 0.87, ...]  (768 numbers)
"queen" → [0.33, -0.09, 0.85, ...]  (nearly the same direction)
"car"   → [-0.7, 0.2, 0.1, ...]     (far from both)
```

Why does this matter?
- The model does arithmetic on vectors, not words. Words have to *become* numbers to flow through the network.
- Because vectors encode semantics, the model can generalize — "king is to queen as uncle is to aunt" becomes vector math.
- **This is the exact building block Week 6 uses for search:** embed your documents, embed the query, find the closest vectors. Same idea, applied to whole chunks of text.

One word of caution: a single token embedding is ambiguous — "bank" is one vector for riverside and institution both. It's the *context* (next section) that disambiguates it.

---

### 4. Transformer Layers: Attention and Context

The embedding flows through a stack of **transformer layers** (e.g., 12–48 of them). Each layer does the same two jobs, and information is refined step by step:

**Job A — Attention: "what should each token look at?"**
Attention lets every token *look at* every other token in the sequence and decide how much weight to give each one. So "bank" next to "river" attends mostly to "river" and means *riverside*; "bank" next to "money" attends to "money" and means *institution*. Context is the whole game.

The intuition behind the mechanism (you don't need the math):
- Each token emits a **query** ("what am I looking for?")
- Each token also carries a **key** ("what do I offer?") and a **value** ("here's my meaning")
- The query of token A is matched against the keys of all tokens; matches determine how much of each value gets mixed into A's new representation.

For a causal LLM, a token can only attend to *itself and earlier tokens* — it cannot peek at the future. That one restriction is the entire difference between a generation model and a reading model, and it's why generation is autoregressive.

**Job B — Feed-forward + fixes:**
After attention, each token passes through a small feed-forward network, plus **residual connections** (add the input back, so deeper layers don't erase earlier knowledge) and **normalization** (keeps numbers stable). You'll hear these names; the intuition is: each layer refines every token's representation, mixing in context, while careful wiring keeps training stable.

**Positional information** is added at the start (each token's embedding gets a position marker), because attention alone has no sense of order — "man bites dog" and "dog bites man" use the same tokens.

**What comes out of the last layer:** a refined vector for every input token, one of which represents the "next token position," ready for scoring.

---

### 5. Scores and Sampling: Choosing the Next Token

The final layer outputs a **score for every token in the vocabulary** (e.g., 50,000 numbers — one per vocab entry). Raw scores are called **logits**. **Softmax** turns them into probabilities that sum to 1. Then we pick:

**Greedy decoding** — always take the highest-probability token. Deterministic, but repetitive and easy to get stuck in loops.

**Sampling** — roll a weighted die over the probabilities. The same prompt can produce different answers. The dials:
- **Temperature** — scales the "peakedness" of the distribution. Low (→ 0): greedy-like, focused. High (> 1): flatter, more random. ~0.7–1.0 is typical for creative generation.
- **top-k** — only consider the k highest-probability tokens, cut the rest.
- **top-p** (nucleus) — keep only the smallest set of tokens whose combined probability reaches p. So p=0.9 means "consider only the top tokens that together carry 90% of the mass."

These aren't a sign of "randomness bug" — sampling is what makes models usable for open-ended generation. Greedy is for cases where you want reproducibility.

```text
logits → softmax → probabilities → temperature/top-k/top-p → pick → append → loop
```

---

### 6. Context and Prompts

Everything the model sees before it starts generating is **context** — your prompt, instructions, and prior turns. The context is fed as tokens; the model attends to all of it.

Modern chat models are shaped so the prompt has a convention (system + user + assistant turns, marked by special tokens):

```text
<|system|> You are a helpful assistant.
<|user|>   Explain LoRA in one sentence.
<|assistant|> LoRA is a cheap way to fine-tune a model...
```

Two consequences you'll feel all course long:
- **The context window is a budget.** Prompt + everything generated counts against it. Longer prompts = more cost, more memory, more latency.
- **Everything in context is conditional on your wording.** Same question, better prompt, better answer. Prompting is a skill, not a chore.

---

### 7. Training vs Inference (the two phases)

| Phase | What Happens | Cost |
|-------|-------------|------|
| **Training** | Model adjusts millions of weights to reduce prediction error over huge data | Expensive, offline |
| **Inference** | Model uses fixed weights to predict the next token on your input | Cheap, online |

**Pre-training** is the expensive once: feed a model hundreds of billions of tokens of text, and for each position, get it to predict the next token; nudge the weights to be less wrong, repeat. After that the model *is* a probability distribution over text — it has "read" a lot and patterns stick in its weights.

**Fine-tuning** (Week 7) is a smaller, cheaper training run on a focused dataset to change *behavior* — teach it your format, tone, or domain. **Prompting and RAG are inference** — no weights change.

You **use** models via inference. The mental line to keep straight:
- Asking it nicely → inference (prompting)
- Giving it your documents to look up → inference (RAG, Week 6)
- Changing its weights on your data → training (fine-tuning, Week 7)

---

### 8. Sizes: What a Model Actually Is

A "7B model" has 7 billion numbers (**parameters** — the weights learned in training). More parameters = more capacity, but each one costs memory and compute. At 16-bit precision each parameter needs ~2 bytes:

| Size | Memory (16-bit) | Fits on |
|------|-----------------|---------|
| 135M | ≈ 0.3 GB | laptop CPU |
| 1.5B | ≈ 3 GB | modest GPU |
| 8B | ≈ 16 GB | serious GPU or quantization |

Add the ~20% overhead for the optimization state at inference, and always leave headroom. This is why the course's "workhorse" is a 135M–1.5B model on a free Colab GPU — you'll feel this trade-off in Week 4's lab when you try to load something too big and hit an out-of-memory error.

---

## Key Takeaway
An LLM predicts the next token, repeatedly. Text → tokens → embeddings → attention → scores → sample. Training learned the weights; inference uses them. If you can explain those six stages, you understand how every LLM works.

---

## Common Pitfalls

### Pitfall 1: Thinking it "understands" like a person
It predicts patterns. Useful, but not truth — that's why outputs need checking (Week 8).

### Pitfall 2: Confusing training and inference
"Can I fine-tune by just asking it to answer well?" No — that's prompting (inference). Fine-tuning is *training* (Week 7).

### Pitfall 3: Tokens ≠ words
Emoji, URLs, and rare words take multiple tokens and eat context. Cost and context limits are measured in tokens, so token count is a real budget you must respect.

### Pitfall 4: Assuming bigger is always better
Bigger models are slower and pricier. A 135M model does a surprising amount with RAG on top (Week 6).

### Pitfall 5: Forgetting the causal mask
A decoder-only LLM can only see the past. It can't "read the whole question first" in one pass — it builds the answer left to right, one token at a time. Understanding this loop explains why outputs take time, why context limits exist, and why greedy decoding can loop.

### Pitfall 6: Over-tweaking sampling settings
Temperature, top-k, top-p are not "more is better" knobs. Start at the model's defaults; only change one dial at a time and observe. Most engineering problems in Weeks 3–5 are data problems, not sampling problems.

---

## Interactive Exercises

1. Open a tokenizer (e.g., https://platform.openai.com/tokenizer or the `tiktoken` library) and count tokens for: a common sentence, an emoji, a URL, and a 100-word paragraph. Note the differences.
2. Copy a 100-word paragraph into any LLM and ask it to predict the "next word" after every sentence break. Does it fit the context each time?
3. Ask the same question twice at temperature 0 and 0.9. Compare determinism vs variety. Then try top-p 0.3 vs 0.95 with temperature fixed.
4. Ask for the same answer in 5 words and in 200 words. Watch how the model keeps generating — the loop doesn't stop until it decides to.
5. In your own words, explain to a friend: why does "bank" mean different things next to "river" vs "money"? (That's attention.)

---

## Quick Reference

```
tokenizer: text  → token IDs          | tokenizer: token IDs → text
embedding: token → vector             | attention: token sees its context
logits → softmax → probabilities      | temperature/top-k/top-p shape them
greedy: always pick the max           | sampling: weighted random pick
loop: append → predict → append       | stop: when a stop token appears
```

## Further Reading
- 3Blue1Brown, "How LLMs work" (visual, ~30 min): https://www.youtube.com/watch?v=LPZh9BOjkQs
- Jay Alammar, "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/
- Hugging Face course, transformer chapter: https://huggingface.co/learn/nlp-course
