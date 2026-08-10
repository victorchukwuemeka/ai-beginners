# Week 2 Assignment: Explain an LLM With Evidence

## Due
End of Week 2

## Objective
Demonstrate you understand tokens, embeddings, and next-token prediction — with actual outputs, not vibes.

## Requirements

### 1. Tokenization evidence (from your lab)
- Show tokenization of 3 texts: a normal sentence, a rare word, an emoji/URL
- Report token counts for each and compare to a rough "4 chars ≈ 1 token" estimate

### 2. Next-token predictions
- For 5 different prompts, print the top-3 next tokens and their probabilities
- Include one prompt where the "right" answer is obvious (e.g., "2 + 2 equals") and one where it's ambiguous
- Write 2–3 sentences interpreting what the probability distribution *means*

### 3. The four stages in your own words (max 300 words)
Explain: tokens → embeddings → attention → sampling. Where does "training" vs "inference" fit? Write it for a curious non-technical friend.

### 4. Sizes table
From the lesson's size table, compute the approximate memory for a 135M, 1.5B, and 8B model at 16-bit (2 bytes/param). Add a sentence: which of these could *you* run today, and on what?

### 5. Reflection (150–300 words)
- What surprised you most about how LLMs actually work?
- What do you think you're missing that Week 3+ will answer?

## Grading
| Criterion | Weight |
|-----------|--------|
| Tokenization evidence correct | 25% |
| Next-token tables + interpretation | 30% |
| Four-stage explanation accurate | 20% |
| Sizes table correct | 10% |
| Reflection is specific | 15% |
