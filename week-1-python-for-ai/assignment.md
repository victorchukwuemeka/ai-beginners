# Week 1 Assignment: JSON In, Models Out

## Due
End of Week 1

## Objective
Prove you can handle the data formats every AI library uses — the prerequisite for everything after.

## Requirements

### 1. Build a 10-example instruction dataset
Create a JSON file with 10 `{"instruction", "output"}` pairs on a topic you care about. Make them varied (short/long, easy/hard).

### 2. A generic filter function
Write `filter_records(records, key, min_len)` that returns records where `len(str(record[key])) >= min_len`. Demonstrate it on your dataset.

### 3. Statistics with comprehensions
For your dataset, report (using comprehensions only, no loops):
- number of records
- longest instruction (length and text)
- how many outputs contain a space

### 4. JSON → JSONL converter
Write a function that converts your JSON list to a JSONL file (one JSON per line), then reads it back and confirms equality.

### 5. Reflection (150–300 words)
- Which part of Python felt unclear? 
- How is an "AI dataset" different from a spreadsheet or a CSV you've used?

## Grading
| Criterion | Weight |
|-----------|--------|
| Dataset built, varied, 10 records | 30% |
| Filter function generic and working | 25% |
| Comprehension stats correct | 20% |
| JSONL converter round-trips | 15% |
| Reflection is specific | 10% |
