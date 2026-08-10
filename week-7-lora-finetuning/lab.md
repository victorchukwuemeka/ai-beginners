# Week 7 Lab: Fine-Tune a Model With LoRA

## Objective
Run a real LoRA fine-tuning experiment on a free Colab GPU, then test whether it actually changed behavior.

## Setup
- Use **Colab with a GPU** (Runtime → Change runtime type → GPU).
- `!pip install transformers datasets peft trl accelerate`

## Tasks

### Task 1: Baseline
Load `HuggingFaceTB/SmolLM2-135M-Instruct` (the base model). Ask it:
`"Answer in exactly 3 bullet points: What are three benefits of exercise?"`
Record the base model's output. Does it follow the "3 bullet points" instruction reliably? (Often no.)

### Task 2: Build a style dataset
Create 30–50 instruction/output pairs where the *output is always exactly 3 bullet points*. Vary the topics (exercise, cooking, python, travel...). Write it as a JSON file with `instruction` + `output` fields.

Examples:
```json
[
  {"instruction": "In exactly 3 bullet points: why do we sleep?", "output": "- Restores the body\n- Consolidates memory\n- Clears brain waste"},
  {"instruction": "In exactly 3 bullet points: benefits of reading.", "output": "- Improves vocabulary\n- Reduces stress\n- Builds empathy"}
]
```

### Task 3: LoRA fine-tune
Use the recipe from the lesson with `r=16`, 3 epochs, batch size 4. Print `trainable_parameters`. Save to `my-lora-adapter`. Note how small the adapter file is vs the base model.

### Task 4: Compare base vs fine-tuned
Load your adapter over the base. Ask **10 fresh prompts** (not in the dataset), all requesting "exactly 3 bullet points". For each, record:
- Does the base model follow the format?
- Does the fine-tuned model follow the format?

### Task 5: Outside the training distribution
Ask the fine-tuned model something totally unrelated (e.g., "Tell me a story"). Did fine-tuning break the base model's abilities? This is your first taste of catastrophic forgetting.

## Deliverable
A notebook `week7_lab.ipynb` + your dataset JSON + the saved adapter. Include the baseline vs fine-tuned comparison table.

## Check Yourself
- [ ] Did trainable params come out under ~1% of total?
- [ ] Did you test on prompts NOT in your training set?
- [ ] Could you load your adapter later without retraining?
