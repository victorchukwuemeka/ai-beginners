# Week 1 Lab: The Python You Actually Need

## Objective
Get comfortable with the exact Python skills you'll use every week: JSON data, lists of dicts, functions, and Colab.

## Tasks

### Task 1: Build an instruction dataset
```python
examples = [
    {"instruction": "What is 2+2?", "output": "4."},
    {"instruction": "Name three colors.", "output": "Red, green, blue."},
    # add 5 more of your own
]
```
Print `len(examples)` and `examples[0]`.

### Task 2: A filtering function
```python
def long_instructions(data, min_len):
    return [e for e in data if len(e["instruction"]) >= min_len]
```
Test it with `min_len=10`. Show the result.

### Task 3: JSON round-trip
Write `examples` to `dataset.json`, reload it, confirm `loaded == examples`.

### Task 4: Work with a JSONL file
Create a small `data.jsonl` (one JSON object per line). Read it back:
```python
records = [json.loads(line) for line in open("data.jsonl")]
```

### Task 5: Colab check
Open Colab, run a cell that prints your name and `2 + 2`. Save the notebook.

## Deliverable
A single notebook `week1_lab.ipynb` with all five tasks.

## Check Yourself
- [ ] Did you print and eyeball every intermediate result?
- [ ] Does your filter function work on empty input (`[]`)?
- [ ] Could you load your JSON back exactly?
