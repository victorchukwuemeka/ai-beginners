# Week 1: Python for AI

## Core Concepts

### Why Python
Every AI framework — PyTorch, Hugging Face, FAISS — speaks Python. You won't write much Python from scratch; you'll *orchestrate* libraries that do the heavy lifting. You need just enough to load data, call functions, and fix errors.

### The Essentials (only what you need)

| Concept | What It Is | Example |
|---------|-----------|---------|
| **Variables** | Store a value | `model_name = "gpt2"` |
| **Lists** | Ordered collection | `labels = ["pos", "neg"]` |
| **Dicts** | Key → value pairs | `tok_settings = {"max_new_tokens": 100}` |
| **Functions** | Reusable block | `def generate(prompt): ...` |
| **Loops** | Repeat over items | `for chunk in chunks:` |
| **Comprehensions** | Build lists in one line | `[len(c) for c in chunks]` |

### Working With Data (the real job)
Most AI work is moving data into the shape a library expects — a list of dicts, a CSV, a JSON file:

```python
import json

# JSON → dict (most Hugging Face datasets are JSON/JSONL)
with open("my_dataset.json") as f:
    dataset = json.load(f)          # list of dicts

# JSONL → list of dicts (one JSON object per line)
records = [json.loads(line) for line in open("data.jsonl")]

# A Hugging Face dataset is basically this:
examples = [
    {"instruction": "What is the capital of France?", "output": "Paris."},
    {"instruction": "Summarize this article.", "output": "..."},
]
```

### Jupyter / Colab
- Cells run one at a time — perfect for experiments.
- **Colab gives a free GPU** — you'll need it in Week 7 (fine-tuning).
- Convention: imports at top, data next, model code, then results.

### The AI Mindset
In traditional code you write exact instructions. In AI code you **load data → call a model → inspect output**. Most of the "programming" is keeping data shaped correctly and reading errors. If you can read a file, build a dict, and call a function, you're ready to run models.

---

## Key Takeaway
You don't need to become a programmer — you need enough Python to *operate* AI libraries. JSON in, lists, dicts, functions, and Colab. That's the whole prerequisite.

---

## Common Pitfalls

### Pitfall 1: Copying code you don't read
**Fix:** Run every line yourself. Change one value and see what breaks.

### Pitfall 2: Shape/dict mismatches
If your list of dicts has different keys, libraries crash. Always print `dataset[0]` and `len(dataset)` first.

### Pitfall 3: Trying to "learn Python fully" first
That's the ML-school trap. Start running models in Week 3 with this tiny base and learn the rest as you hit it.

---

## Interactive Exercises

1. Build a list of 5 dicts (e.g., `{"task": "...", "output": "..."}`).
2. Write a function that returns only tasks containing a keyword.
3. Load any JSON/JSONL file and print its first record.
4. Use a comprehension to list the lengths of all your instructions.
5. Run your code in Colab (create a free account) — this is your workspace for the course.

---

## Quick Reference

```python
len(x)                 # size
x[0]                   # first item
x[-1]                  # last item
x.append(v)            # add to list
x.keys() / x.values()  # dict access
json.load(f)           # read JSON
json.dump(d, f)        # write JSON
[expr for x in lst]    # comprehension
```

---

## Further Reading
- Colab intro: https://colab.research.google.com/notebooks/intro.ipynb
- Python for Everybody (free, short): https://www.py4e.com
- Hugging Face course (starts right here, at your level): https://huggingface.co/learn/nlp-course
