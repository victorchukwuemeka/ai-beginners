# Week 7 Assignment: A Style-Shifting Fine-Tune With Evidence

## Due
End of Week 7

## Objective
Fine-tune a model to a specific behavior and **prove the change with a comparison, not vibes**.

## Requirements

### 1. Choose a target behavior
Pick something a base model won't reliably do. Ideas:
- Always respond in exactly 3 bullet points
- Always end with a one-word question
- A specific persona ("you are a grumpy sysadmin")
- Always output valid JSON with fixed fields

### 2. Dataset (150+ samples)
Create a JSON dataset with varied inputs for your target behavior. 150–300 clean pairs. Vary topics/wordings so the model learns the *rule*, not your exact sentences.

### 3. LoRA fine-tune
- Use a small model (SmolLM2-135M/360M or GPT-2) so it fits free Colab
- `r=16`, 3 epochs
- Show `print_trainable_parameters()` output in your notebook

### 4. The comparison (the heart)
Build a **held-out test set of 20 prompts** the model never saw, each needing your behavior. Run all 20 through **base** and **fine-tuned** models. Score each output:
- Behavior followed? (yes/no)
- Correct content? (yes/no)

Report a table with both models' scores. This is your first real **evaluation**: fine-tuned vs baseline on unseen data.

### 5. Forgetting check
Test the fine-tuned model on 5 generic tasks (summarize, joke, math). Note any degradation vs base.

### 6. Reflection (150–300 words)
- Did LoRA work? By how much, measured how?
- What would you change in the dataset if you iterated?
- When would you NOT fine-tune and use RAG/prompting instead?

## Grading
| Criterion | Weight |
|-----------|--------|
| Dataset quality (150+, varied, clean) | 25% |
| Fine-tune runs correctly, adapter saved | 20% |
| Base-vs-tuned comparison on held-out test set | 30% |
| Forgetting check | 10% |
| Reflection is specific and honest | 15% |
