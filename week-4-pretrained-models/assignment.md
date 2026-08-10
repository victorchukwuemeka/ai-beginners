# Week 4 Assignment: Own the Machinery

## Due
End of Week 4

## Objective
Demonstrate you can load, inspect, and drive the tokenizer and model directly — without `pipeline()`.

## Requirements

### 1. Tokenizer deep-dive
- Show tokenization of 4 texts: normal sentence, rare word, emoji, URL — with token counts
- Show `vocab_size` of GPT-2 and SmolLM2-135M
- Show a **padded batch** of 3 prompts: `input_ids.shape` and the `attention_mask`. Explain in one sentence what the mask tells the model.

### 2. Logits reading
For a prompt of your choice:
- Print logits shape
- Softmax the last position
- Show top-5 tokens with probabilities
- Interpret: what does the probability distribution tell you about the model's confidence?

### 3. Memory table
Load two models (e.g., GPT-2 and SmolLM2-135M) and record for each:
`parameters | memory footprint (MB) | memory per param (bytes) | my machine: CPU or GPU?`

### 4. Generate without pipeline
Use tokenizer + `AutoModelForCausalLM` + `generate()` directly to produce text from a seed. Include it in the notebook with the settings you chose.

### 5. Reflection (150–300 words)
- What does `pipeline()` hide that you now see?
- Why do you think batching exists (what's the cost of the padded "hi" example)?

## Grading
| Criterion | Weight |
|-----------|--------|
| Tokenizer tasks complete and correct | 25% |
| Logits/softmax interpretation | 25% |
| Memory table correct | 20% |
| Manual generate works | 20% |
| Reflection is specific | 10% |
