# LLM Prompt Optimizer (lightweight version)

A smaller, standalone tool for checking token usage and tightening up prompts — built separately from my bigger final-year prompt optimizer project, more of a quick utility/library.

## Features
- Token counting via `tiktoken` (GPT-3.5/4/4o tokenizers)
- Flags filler phrases that waste tokens
- Basic clarity check — flags prompts missing a clear output directive
- Rewrites the prompt and reports token savings
- Batch mode to compare multiple prompt variants at once
- Rough cost estimate based on token count

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Using it as a library

```python
from optimizer import analyze, report

result = analyze("I would like you to please summarize the following text in order to provide a brief overview.")
print(report(result))
# Token savings: 12 tokens (34.3% reduction)
# Optimized: "Summarize the following text briefly."
```

```python
from optimizer import batch_analyze

prompts = ["Can you please help me...", "Generate a list of..."]
results = batch_analyze(prompts)
```

## Stack
Python, tiktoken, Streamlit, regex

## Notes
- Redundancy detection is just a list of common filler phrases, not a learned model
- Clarity scoring is rule-based, would need more cases to catch subtler issues
