# Week 1: Python for AI

## Core Concepts

### Why Python
Every AI framework — PyTorch, Hugging Face, FAISS — speaks Python. You won't write much Python from scratch; you'll *orchestrate* libraries that do the heavy lifting. You need just enough to load data, call functions, and fix errors.

Your job in this course is not to become a software engineer. It's to become fluent enough that when a model errors on malformed data, you can see it, fix the data, and re-run. That skill — shaping data and reading errors — is 80% of applied AI work.

---

### 0. Setup: Your Toolkit (do this once)

You have two ways to run Python in this course:

| Path | Install effort | GPU | Best for |
|------|----------------|-----|----------|
| **Google Colab** | none — just a browser | free T4 | Weeks 1–2, anything needing a GPU |
| **Local Jupyter** | ~15 min, one time | your own machine | offline work, real datasets, Week 7+ |

Weeks 1–2 run fine on Colab alone. Do the local setup when convenient — you'll want it eventually.

#### Path 1: Colab (zero install)

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with any Google account.
2. **File → New notebook.** You are now running Python.
3. Free GPU (you'll need it in Week 7): **Runtime → Change runtime type → T4 GPU**.
4. Confirm the GPU is real:

```python
import torch
print(torch.cuda.is_available())   # True = GPU attached
```

Nothing to download, nothing to break. Sessions die when idle (~90 min), so save notebooks to Drive (**File → Save a copy in Drive**).

#### Path 2: Local Jupyter

**Step 1 — Install Python** (any version 3.10+ works)

**Windows:**
- Download the installer from [python.org/downloads](https://www.python.org/downloads/). On the first screen, **check "Add python.exe to PATH"** — skipping this checkbox is the #1 Windows setup mistake.
- Or, in PowerShell: `winget install Python.Python.3.12`
- Verify (open a *new* PowerShell window): `python --version`

**macOS:**
- With Homebrew: `brew install python` — or grab the python.org installer.
- Verify in Terminal: `python3 --version`

**Linux (Ubuntu/Debian):**

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
python3 --version
```

(Fedora/RHEL: `sudo dnf install python3 python3-pip`)

**Step 2 — Make a work folder with a virtual environment**

A venv keeps this course's packages isolated from your system Python — when Week 7 wants a specific library version, it won't wreck anything else.

```bash
mkdir ai-course && cd ai-course
python3 -m venv .venv           # Windows: python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

Your prompt now starts with `(.venv)` — packages install into it, not your system. Re-run the activate line in every new terminal session; leave with `deactivate`.

**Step 3 — Install and launch Jupyter**

```bash
pip install jupyterlab
jupyter lab        # opens http://localhost:8888 in your browser
```

Prefer the classic look? `pip install notebook`, then `jupyter notebook`.

**Step 4 — Verify everything**

```python
import sys, json
print(sys.version)      # 3.x.y — any 3.10+ is fine
print("setup ok")
```

**Optional but worth it — VS Code:** install [VS Code](https://code.visualstudio.com/) plus its Python extension, then **File → Open Folder** on your work folder and **Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`**. It opens `.ipynb` notebooks natively — Jupyter with autocomplete and a real debugger.

**Troubleshooting quick hits:**

| Symptom | Fix |
|---------|-----|
| `python: command not found` (Windows) | reinstall with "Add to PATH" checked, or try `py --version` |
| `pip: command not found` | use `python -m pip install ...` instead |
| `jupyter: command not found` | venv isn't activated — run the activate line again |
| macOS pops up Xcode tools | safe to let it install, or use the python.org build |

---

### 1. Variables and Types

A variable is just a name bound to a value. Every value has a **type**, and the type decides what you can do with it.

| Type | Meaning | Example |
|------|---------|---------|
| `int` | whole number | `n = 42` |
| `float` | decimal | `temp = 0.9` |
| `str` | text | `name = "gpt2"` |
| `bool` | True/False | `is_cuda = True` |
| `list` | ordered, mutable | `labels = ["pos", "neg"]` |
| `dict` | key → value | `cfg = {"max_new_tokens": 100}` |
| `tuple` | ordered, immutable | `shape = (3, 512)` |
| `set` | unique items | `tags = {"rag", "lora"}` |
| `None` | "nothing" | `result = None` |

Always check your assumptions: `type(x)` tells you what you're holding, and `str(x)` converts almost anything to text for printing.

```python
config = {"max_new_tokens": 100, "temperature": 0.7}
print(type(config))          # <class 'dict'>
print(len(config))           # 2
```

**The type errors you'll see all the time:** trying to add a `str` to an `int`, calling `.lower()` on a number, or indexing a value that's `None`. When in doubt, `print(type(x))`.

---

### 2. Strings

Model input and output are strings. You'll slice, search, and format them constantly.

```python
prompt = "summarize this article"

prompt.upper()          # "SUMMARIZE THIS ARTICLE"
prompt.split()          # ["summarize", "this", "article"]
prompt.startswith("sum")  # True
len(prompt)             # 23
prompt[0]               # "s"
prompt[-3:]             # "cle"
"rag" in prompt         # False
```

**F-strings** are how you build prompts dynamically — you will use these every week:

```python
topic = "LoRA"
prompt = f"Explain {topic} to a beginner in 3 sentences."
print(prompt)
# Explain LoRA to a beginner in 3 sentences.
```

`str.format()` and `%` both work but f-strings are the modern, readable choice.

---

### 3. Lists

An ordered, changeable collection — the workhorse of data handling.

```python
labels = ["pos", "neg", "neutral"]

labels[0]        # "pos"      (indexing starts at 0)
labels[-1]       # "neutral"  (negative index = from the end)
labels[1:]       # ["neg", "neutral"]
labels.append("other")   # add to the end
len(labels)      # 4
"pos" in labels  # True
```

Nested data is everywhere in AI: a list of dicts is literally the shape of a Hugging Face dataset.

```python
batch = [
    {"text": "Great movie",   "label": 1},
    {"text": "Terrible film", "label": 0},
]
batch[1]["text"]        # "Terrible film"
```

**Slicing** — `lst[start:stop]` — returns a new list; `lst[:n]` takes the first n items, `lst[-n:]` the last n. You'll use it constantly for previewing data (`print(data[:3])`).

---

### 4. Dictionaries

Key → value pairs. Most AI configuration — model settings, tokenizer settings, training args — lives in dicts.

```python
cfg = {"max_new_tokens": 100, "temperature": 0.7, "do_sample": True}

cfg["temperature"]          # 0.7
cfg.get("top_p", 0.0)       # 0.0  — .get with a default, won't crash
cfg.keys()                  # dict_keys(['max_new_tokens', ...])
cfg.values()                # dict_values([100, 0.7, True])
cfg["temperature"] = 1.0    # update a value
cfg["top_p"] = 0.95         # add a new key
"top_p" in cfg              # True
```

**`.get()` is your friend.** `cfg["missing"]` raises `KeyError`; `cfg.get("missing", 0.0)` returns the default. In AI pipelines, missing keys are common — write defensive code.

---

### 5. Control Flow

**Conditionals:**

```python
score = 0.85
if score > 0.8:
    print("good")
elif score > 0.5:
    print("ok")
else:
    print("needs work")
```

**Loops** — two patterns cover 95% of what you'll need:

```python
# Loop over items
for label in labels:
    print(label)

# Loop with a counter
for i, example in enumerate(dataset):
    print(i, example["instruction"])

# Loop over two lists at once
for text, label in zip(texts, labels):
    print(text, "->", label)
```

`range(n)` gives `0..n-1` — use it when you genuinely need index math, not to re-invent `for x in items`.

**A note on style:** dicts and lists are copied by reference, not value.

```python
a = [1, 2, 3]
b = a          # b IS a, same list
b.append(4)
print(a)       # [1, 2, 3, 4]  — surprising!
c = a.copy()   # a real copy
```

When you pass a list to a function and mutate it inside, the caller's list changes too. `copy()` (or `list(a)`) protects you.

---

### 6. Functions

A function is a reusable block that takes inputs and returns an output.

```python
def is_long(text, min_len=50):
    return len(text) >= min_len

is_long("hello")            # False
is_long("hello", min_len=3) # True
```

Rules that will save you pain:
- **Default arguments** make a function reusable without breaking old calls.
- **Return early, return often** — compute a value and `return` it rather than printing.
- **One job per function** — filter, then score, then format. Compose small pieces.
- **`lambda`** for one-liners passed to other functions:

```python
records.sort(key=lambda r: len(r["output"]))   # sort by output length
```

- **`*args` / `**kwargs`** — you'll mostly *see* these in library signatures. They mean "any number of extra positional/keyword arguments," which is how libraries accept flexible options.

```python
def generate(prompt, **kwargs):
    print("extra options:", kwargs)

generate("hi", temperature=0.9, max_new_tokens=50)
```

---

### 7. Comprehensions

Python's superpower for building collections in one readable line. A comprehension has four parts:

```
[  expression  for  item  in  iterable  if  condition  ]
```

```python
chunks = ["a", "bb", "ccc"]

lengths = [len(c) for c in chunks]          # [1, 2, 3]
long_ones = [c for c in chunks if len(c) > 1]  # ["bb", "ccc"]
```

**Dict comprehensions** are just as common — you'll use them to reshape data:

```python
texts = ["good", "bad", "ok"]
lengths = {t: len(t) for t in texts}
# {"good": 4, "bad": 3, "ok": 2}
```

Read a comprehension as: "build a list where each item is `expression`, computed by running `for item in iterable`, skipping anything that fails `if condition`." If a comprehension gets long, a plain loop is fine — clarity wins.

---

### 8. Working With Data (the real job)

Most AI work is moving data into the shape a library expects — a list of dicts, a CSV, a JSON file. This is the week's most important section.

**JSON** is the universal data format for AI — models, datasets, and APIs all speak it. It maps directly onto Python types: objects ↔ dicts, arrays ↔ lists, strings ↔ strings, numbers ↔ ints/floats, `true/false` ↔ `True/False`.

```python
import json

# dict/list → JSON string
s = json.dumps({"task": "classify", "k": 2}, indent=2)
print(s)

# JSON string → dict/list
back = json.loads(s)

# dict/list → file
with open("my_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

# file → dict/list  (most HF datasets are JSON/JSONL)
with open("my_dataset.json") as f:
    dataset = json.load(f)          # list of dicts
```

**JSONL** — one JSON object per line — is the standard for large datasets because each line can be streamed independently without loading the whole file. (This is what Hugging Face's `datasets` library uses under the hood.)

```python
# write JSONL: one dict per line
with open("data.jsonl", "w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

# read JSONL: a list of dicts, one per line
records = [json.loads(line) for line in open("data.jsonl")]
```

**Robust JSONL** — real files have blank lines, truncated downloads, and garbage rows. Skip what you can't parse, and count what survived:

```python
records, bad_lines = [], []
with open("data.jsonl") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines.append(i)

print(f"{len(records)} ok, {len(bad_lines)} bad: {bad_lines[:5]}")
```

Never trust a load silently — always compare `len(records)` against what you expected.

**Nested JSON** — API responses and model outputs are dicts inside lists inside dicts. Read access chains left to right: "into `usage`, then `prompt_tokens`."

```python
resp = {
    "model": "gpt2",
    "usage": {"prompt_tokens": 5, "completion_tokens": 12},
    "choices": [{"text": "Paris is...", "finish_reason": "stop"}],
}

resp["model"]                        # "gpt2"
resp["usage"]["completion_tokens"]   # 12
resp["choices"][0]["text"]           # "Paris is..."
resp.get("missing", {}).get("key")   # None — safe chain, no crash
```

Lost three levels deep? Stop and print one level at a time: `print(resp["usage"])`, then go deeper.

**Python ↔ JSON translation** — six types, two surprises:

| Python | JSON | Watch out |
|--------|------|-----------|
| `dict` | object | keys become strings |
| `list` | array | |
| `tuple` | array | **round-trips into a list** |
| `str` | string | |
| `int` / `float` | number | |
| `True` / `False` | `true` / `false` | case changes |
| `None` | `null` | |

```python
t = (1, 2)
back = json.loads(json.dumps(t))
print(type(back))     # <class 'list'> — your tuple didn't survive
```

Some things don't translate at all — `set`, `datetime`, and your own classes raise `TypeError` on `dumps`. Convert them to strings/lists first.

**Handy `dumps` options:**

```python
json.dumps(cfg, indent=2)             # human-readable — for files you'll eyeball
json.dumps(cfg, sort_keys=True)       # stable order — clean diffs, clean git
json.dumps(cfg, ensure_ascii=False)   # keeps "café" as café, not caf\u00e9
```

**When JSON breaks** — the error is `json.JSONDecodeError`, and its message names the exact line and column. The usual culprits:

1. The file is really JSONL — many objects, one per line. Parse line by line.
2. Truncated download — the tail is cut off mid-object.
3. Python-style data pasted as JSON — single quotes and trailing commas are invalid JSON.

```python
bad = "{'task': 'classify'}"      # single quotes: valid Python, invalid JSON
try:
    json.loads(bad)
except json.JSONDecodeError as e:
    print(e)                       # Expecting property name enclosed in double quotes...
```

Rule of thumb: if `json.load` fails on a multi-line file, try the JSONL pattern before anything else.

**Brain teasers** — attempt each before peeking at the answers:

1. Flatten: `data = [{"tags": ["a", "b"]}, {"tags": ["c"]}, {}]` → `["a", "b", "c"]` in one comprehension.
2. `x = (1, 2)` goes through `json.loads(json.dumps(x))`. What type comes back, and what could break downstream?
3. A 2 GB JSONL file won't fit in RAM. Why does the line-by-line pattern still work?
4. `cfg = json.load(open("cfg.json"))` returns `{"temperature": "0.7"}`. What happens later at `if cfg["temperature"] > 0.8:` — and where should the fix live?
5. Merge `defaults` and `user_cfg` into one dict where user values win — one expression, no loop.

<details><summary>Answers</summary>

1. `[t for d in data for t in d.get("tags", [])]` — the `.get` handles the record with no `tags` key.
2. A `list`. Anything that relied on tuple behavior (hashability, use as a dict key) behaves differently.
3. Only one line is ever held as a string at a time — parse it, keep what you need, move on. Memory grows with kept records, not file size.
4. `TypeError: '>' not supported between instances of 'str' and 'int'` — raised far away from the real cause. Fix at the boundary: `cfg["temperature"] = float(cfg["temperature"])` right after loading.
5. `{**defaults, **user_cfg}` — later keys overwrite earlier ones.

</details>

---

**The canonical AI dataset shape** — this exact structure will look familiar in Weeks 3, 6, and 7:

```python
examples = [
    {"instruction": "What is the capital of France?", "output": "Paris."},
    {"instruction": "Summarize this article.", "output": "..."},
]
```

**CSV** — you may import CSV data (from spreadsheets). The stdlib `csv` module handles escaping for you; don't parse it by splitting on commas:

```python
import csv
with open("data.csv", newline="") as f:
    rows = list(csv.DictReader(f))   # list of dicts
```

**First thing you do with any dataset — peek at it:**

```python
print(type(dataset))     # list or dict?
print(len(dataset))      # how many?
print(dataset[0])        # what does one record look like?
```

Errors later weeks come overwhelmingly from skipping this step.

---

### 9. Errors and Debugging

You will hit errors constantly. Learning to read them calmly is the single most valuable Python skill for this course.

**Anatomy of a traceback** — read bottom-up. The last line is the error *type* and message; the lines above show the call chain:

```
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-3-...> in <module>()
----> 1 print(record["text"])
KeyError: 'text'
```

**The five errors you'll actually see:**

| Error | Meaning | Fix |
|-------|---------|-----|
| `NameError: name 'x' is not defined` | typo or forgot to assign | check spelling / assignment |
| `TypeError: ... unsupported operand` | mixing types (str + int) | `int(x)` / `str(x)` |
| `KeyError: 'text'` | key missing from dict | use `.get()` or inspect the dict |
| `IndexError: list index out of range` | index past the end | check `len()` / use `.get()` |
| `ValueError: ...` | wrong value for type | check the value, e.g. `int("abc")` |

**Debugging strategy — the 3 prints:**
1. `print(type(x))` — what am I actually holding?
2. `print(len(x))` — how many?
3. `print(x[:2])` — what does the start look like?

Then shrink the problem: if a comprehension fails, test it on one item. If a function fails, call it with a known input.

**`try/except`** — mostly for handling expected failures gracefully (e.g., a missing file or network hiccup):

```python
try:
    data = json.load(open("missing.json"))
except FileNotFoundError:
    print("using fallback data")
    data = []
```

Don't wrap everything in `try/except` — for this course, letting errors crash loudly and reading the traceback is the faster path.

---

### Debug Gym — fix these before moving on

Reading about errors is not the same as reading errors. Run each snippet, read the traceback bottom-up, say the fix out loud, then apply it. Five minutes here beats an hour of re-reading.

**Bug 1 — KeyError:**

```python
record = {"instruction": "Summarize this.", "output": "Paris."}
print(record["prompt"])
```

Fix it twice: correct the key, then rewrite with `.get("prompt", "")`.

**Bug 2 — the type that lied:**

```python
scores = ["0.9", "0.4", "0.75"]     # came from a CSV — all strings
print(max(scores) > 0.8)
```

Two bugs hide here: the comparison crashes, and even without it, `max()` on strings compares alphabetically, not numerically. Fix until the answer is `True`.

**Bug 3 — aliasing:**

```python
base_cfg = {"temperature": 0.7, "top_p": 0.9}
run_a = base_cfg
run_a["temperature"] = 1.0
print(base_cfg["temperature"])      # why 1.0?
```

Make `run_a` independent, then verify `base_cfg` still prints `0.7`.

**Bug 4 — IndexError:**

```python
batch = [{"text": "hi"}, {"text": "yo"}]
for i in range(3):
    print(batch[i]["text"])
```

Make the loop survive without shrinking `range(3)` — guard with `len(batch)`.

**Bug 5 — ValueError:**

```python
k = int("top_3")
```

Where's this in real life? `int(cfg.get("k", "3"))` when a config value isn't numeric. Make it not crash with a sensible default.

**Final boss — messy data:** build this file, then load it with the robust JSONL pattern from §8 and report: how many lines parsed, which were skipped, and which parsed records are missing the `output` key.

```python
rows = [
    '{"instruction": "Q1", "output": "A1"}',
    '{"instruction": "Q2"}',
    'not json at all',
    '',
    '{"instruction": "Q3", "output": "A3"}',
]
with open("messy.jsonl", "w") as f:
    f.write("\n".join(rows))
```

Expected outcome: 3 parsed, 2 skipped, 1 record missing `output`. If your numbers differ, reread the traceback — not the code.

---

### 10. Jupyter / Colab

- **Cells run one at a time** — perfect for experiments. `Shift+Enter` runs the current cell.
- **Markdown cells** let you document between code cells — write your reasoning as you go.
- **Convention:** imports at top, then data, then model code, then results.
- **Colab gives a free GPU** — you'll need it in Week 7 (fine-tuning). Enable it via **Runtime → Change runtime type → T4 GPU**.
- **`!` runs shell commands** — `!pip install transformers` works directly in a cell.
- **`?` gives help** — `pipeline?` shows the docstring; `pipeline??` shows the source.
- **Magic commands** you'll use: `%time` (timing), `%load` (load a file into a cell), `%whos` (list variables).
- **Check the GPU** before trusting it:

```python
import torch
print(torch.cuda.is_available())   # True = you're on a GPU
```

- **State persists per session**: variables survive across cells, but the kernel state is lost when the session ends. Save datasets to files and models get re-downloaded as needed.

---

### The AI Mindset

In traditional code you write exact instructions. In AI code you **load data → call a model → inspect output**. Most of the "programming" is keeping data shaped correctly and reading errors.

The flow you'll repeat for 9 more weeks:

```python
data = load(...)        # 1. get data in the right shape
prompts = [format(x) for x in data]   # 2. reshape it
outputs = [model(p) for p in prompts] # 3. call the model
score(outputs)          # 4. inspect the result
```

If you can read a file, build a dict, call a function, and read a traceback, you're ready to run models.

---

## Key Takeaway
You don't need to become a programmer — you need enough Python to *operate* AI libraries. JSON in, lists of dicts, functions, comprehensions, tracebacks, and Colab. That's the whole prerequisite.

---

## Common Pitfalls

### Pitfall 1: Copying code you don't read
**Fix:** Run every line yourself. Change one value and see what breaks. Typing it yourself is how the mental model forms.

### Pitfall 2: Shape/dict mismatches
If your list of dicts has different keys, libraries crash. Always print `dataset[0]` and `len(dataset)` first. Then check that *every* record has the key you expect, not just the first one.

### Pitfall 3: Mutable aliasing
`b = a` does not copy a list. You mutate `b`, you've mutated `a`. Use `.copy()` (or `copy.deepcopy()` for nested structures).

### Pitfall 4: Mixing types without knowing it
JSON gives you numbers as `int`/`float`; CSV gives you everything as strings. `len(str(x))` saves more bugs than any other trick.

### Pitfall 5: Trying to "learn Python fully" first
That's the ML-school trap. Start running models in Week 3 with this tiny base and learn the rest as you hit it.

---

## Interactive Exercises

1. Build a list of 5 dicts (e.g., `{"task": "...", "output": "..."}`).
2. Write a function that returns only tasks containing a keyword.
3. Load any JSON/JSONL file and print its first record.
4. Use a comprehension to list the lengths of all your instructions, then a dict comprehension mapping each instruction to its length.
5. Deliberately trigger each of the five common errors (KeyError, IndexError, TypeError, ValueError, NameError) and read its traceback out loud.
6. In Colab, run `!nvidia-smi` (or check `torch.cuda.is_available()`) to confirm whether you have a GPU.

---

## Quick Reference

```python
type(x)                 # what type is x?
str(x) / int(x) / float(x)  # conversions
len(x)                  # size
x[0] / x[-1]            # first / last item
x[:3]                   # slice: first 3
x.append(v)             # add to list
x.get(k, default)       # safe dict access
f"{name}"               # f-string formatting
[e for e in lst if cond]   # list comprehension
{ k: v for ... }        # dict comprehension
json.load(f)            # read JSON file → dict/list
json.loads(s)           # JSON string → dict/list
json.dump(d, f)         # dict/list → JSON file
json.dumps(d)           # dict/list → JSON string
sorted(lst, key=lambda x: ...)  # sort by a key
```
