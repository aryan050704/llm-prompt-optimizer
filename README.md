# LLM Prompt Optimizer

Analyze and optimize LLM prompts for **token efficiency**, redundancy, and clarity. Includes a Streamlit dashboard and a Python library you can import directly.

## Features
- Token counting via `tiktoken` (supports GPT-3.5, GPT-4, GPT-4o tokenizers)
- Redundancy detection — flags filler phrases that waste tokens
- Clarity scoring — checks for missing output directives
- Automatic prompt optimization with token savings report
- Batch analysis for comparing multiple prompt variants
- Live cost estimation

## Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use as a Library

```python
from optimizer import analyze, report

result = analyze("I would like you to please summarize the following text in order to provide a brief overview.")
print(report(result))
# Token savings: 12 tokens (34.3% reduction)
# Optimized: "Summarize the following text briefly."
```

## Batch Analysis

```python
from optimizer import batch_analyze

prompts = ["Can you please help me...", "Generate a list of..."]
results = batch_analyze(prompts)
```

## Tech Stack
`Python` `tiktoken` `Streamlit` `Regex`
