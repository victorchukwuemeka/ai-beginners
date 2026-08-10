# Week 7: Fine-Tuning with LoRA

## Core Concepts

### What Fine-Tuning Is
Fine-tuning = continue training a pretrained model on *your* data so it behaves the way you want. This is Week 3's gradient-descent loop applied to a real, huge model.

### When to Use What (the decision framework)

| Situation | Best Approach |
|-----------|--------------|
| Need the model to use your live, changing documents | **RAG** (Week 6) |
| Need the model to know facts that *never change* | Fine-tuning (or RAG too) |
| Need a specific **format/tone/behavior** (JSON output, your brand voice, chat style) | **Fine-tuning** |
| Prompting already works well enough | Don't touch the model |

Rule of thumb: **RAG is for facts, fine-tuning is for form.** Most production systems use both.

### The Problem With Full Fine-Tuning
Updating all 8 billion weights needs enormous GPU memory (a 70B model can need 500 GB+). Full fine-tuning is usually overkill and risky (catastrophic forgetting — the model forgets what it knew).

### LoRA: The Cheap Way
**LoRA (Low-Rank Adaptation)** doesn't train the original weights at all. It:
1. Freezes the model.
2. Adds small "adapter" matrices at each layer.
3. Trains *only* the adapters (often <1% of parameters).

Result: ~95%+ of full fine-tuning quality at a fraction of the cost and memory. In 2026, LoRA/QLoRA cover the vast majority of fine-tuning needs.

```
Original weight W (frozen)          LoRA: W' = W + BA
   [3000 × 4000]                      A: [rank × 4000], B: [3000 × rank]
                                      rank ≈ 8–16 → tiny training set
```

### Dataset Formats
Instruction tuning data is (mostly) triples:

```json
[
  {"instruction": "What is the capital of France?", "output": "Paris."},
  {"instruction": "Summarize: ...", "output": "..."}
]
```

Or conversation format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Quality beats quantity.** 500 clean, varied examples beat 5,000 noisy ones.

### The Recipe (SmolLM2 on Colab, free GPU)

```bash
pip install transformers datasets peft accelerate bitsandbytes
```

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
tok   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
tok.pad_token = tok.eos_token

dataset = load_dataset("json", data_files="my_dataset.json")["train"]

lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # "< 1% trainable"

trainer = SFTTrainer(
    model=model, train_dataset=dataset,
    args=TrainingArguments(output_dir="lora_out", num_train_epochs=3,
                           per_device_train_batch_size=4, logging_steps=10),
)
trainer.train()
model.save_pretrained("my-lora-adapter")   # tiny file, not the whole model
```

That's the whole thing: load model → wrap with LoRA → train → save the small adapter.

### Loading Your Adapter Later
```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = PeftModel.from_pretrained(base, "my-lora-adapter")
```
The adapter is a few MB, even though the base model is ~500 MB. This is what makes fine-tuning accessible to beginners.

---

## Key Takeaway
Fine-tuning adapts a model's behavior. **LoRA trains tiny adapter matrices instead of the whole model**, making it cheap enough to do on a free Colab GPU. Use RAG for changing facts, LoRA for form/behavior. Save only the adapter.

---

## Common Pitfalls

### Pitfall 1: Too-small or one-note dataset
Model learns to parrot your 3 examples. Aim for 200–1,000 varied samples.

### Pitfall 2: Forgetting to set a pad token
GPT-2/SmolLM often lack one. `tok.pad_token = tok.eos_token`.

### Pitfall 3: Training until loss = 0
That's memorization, not learning. Watch for the eval loss going back up (overfitting).

### Pitfall 4: Judging success by training loss alone
You must compare the fine-tuned model against the base model on **unseen** prompts. That's Week 8.

### Pitfall 5: Using LoRA when you should use RAG
If your data changes weekly, fine-tuning is the wrong tool — RAG is.

---

## Interactive Exercises

1. Take a prompt that the base SmolLM2 answers generically. Note it.
2. Write 30–50 instruction pairs in a specific style (e.g., "always answer in exactly 3 bullet points").
3. Fine-tune with LoRA (5–10 min on Colab GPU).
4. Ask the same prompt to base vs fine-tuned. Compare.
5. Check `model.print_trainable_parameters()` — what % did you actually train?

---

## Quick Reference

```python
get_peft_model(model, LoraConfig(r=16, lora_alpha=32))  # wrap
SFTTrainer(...).train()                                  # train
model.save_pretrained("adapter")                         # save small adapter
PeftModel.from_pretrained(base, "adapter")               # reload
```

---

## Further Reading
- LoRA paper: https://arxiv.org/abs/2106.09685
- HF PEFT docs: https://huggingface.co/docs/peft
- SmolLM2 (great for beginners): https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
