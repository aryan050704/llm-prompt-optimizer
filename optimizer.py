"""
LLM Prompt Optimizer — measures and improves prompt token efficiency.
Supports OpenAI-compatible APIs and local Ollama models.
"""
import re
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import tiktoken


@dataclass
class PromptAnalysis:
    original_prompt: str
    token_count: int
    char_count: int
    redundancy_score: float       # 0-1, higher = more redundant
    clarity_score: float          # 0-1, higher = clearer
    suggestions: list[str] = field(default_factory=list)
    optimized_prompt: Optional[str] = None
    optimized_token_count: Optional[int] = None
    token_savings: Optional[int] = None
    savings_pct: Optional[float] = None


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# Patterns that inflate token count without adding meaning
_FILLER_PATTERNS = [
    (r"\bplease\b\s*", ""),
    (r"\bkindly\b\s*", ""),
    (r"\bcould you\b\s*", ""),
    (r"\bI would like you to\b\s*", ""),
    (r"\bI want you to\b\s*", ""),
    (r"\bCan you please\b\s*", ""),
    (r"\bAs an AI language model,?\s*", ""),
    (r"\bNote that\b\s*", ""),
    (r"\bit is important to note that\b\s*", "", re.IGNORECASE),
    (r"\bIt should be noted that\b\s*", "", re.IGNORECASE),
    (r"\s{2,}", " "),
]

_REDUNDANCY_MARKERS = [
    r"\b(very|really|quite|extremely|absolutely)\b",
    r"\b(in order to)\b",
    r"\b(due to the fact that)\b",
    r"\b(at this point in time)\b",
    r"\b(for the purpose of)\b",
]


def _redundancy_score(text: str) -> float:
    words = len(text.split())
    if words == 0:
        return 0.0
    hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in _REDUNDANCY_MARKERS)
    filler_hits = sum(
        len(re.findall(p[0], text, re.IGNORECASE if len(p) > 2 else 0))
        for p in _FILLER_PATTERNS if p[1] == ""
    )
    return min(1.0, (hits + filler_hits * 0.5) / (words / 10))


def _clarity_score(text: str) -> float:
    score = 1.0
    sentences = re.split(r"[.!?]", text)
    avg_sentence_len = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
    if avg_sentence_len > 30:
        score -= 0.2
    if not re.search(r"\b(output|format|return|respond|answer|list|explain|summarize|classify|generate)\b", text, re.IGNORECASE):
        score -= 0.15
    if len(text.split()) < 5:
        score -= 0.3
    return max(0.0, score)


def _build_suggestions(text: str, redundancy: float, clarity: float) -> list[str]:
    suggestions = []
    if redundancy > 0.2:
        suggestions.append("Remove filler phrases (e.g. 'please', 'could you', 'I would like you to').")
    if clarity < 0.7:
        suggestions.append("Add a clear output directive (e.g. 'Return a JSON list', 'Explain in 3 bullet points').")
    if re.search(r"\b(in order to)\b", text, re.IGNORECASE):
        suggestions.append("Replace 'in order to' with 'to'.")
    if re.search(r"\b(due to the fact that)\b", text, re.IGNORECASE):
        suggestions.append("Replace 'due to the fact that' with 'because'.")
    if len(text.split()) > 200:
        suggestions.append("Consider splitting into a system prompt and a shorter user prompt.")
    if not suggestions:
        suggestions.append("Prompt looks efficient. Consider adding few-shot examples for better accuracy.")
    return suggestions


def _apply_optimizations(text: str) -> str:
    result = text
    replacements = [
        (r"\bI would like you to\b\s*", "", re.IGNORECASE),
        (r"\bI want you to\b\s*", "", re.IGNORECASE),
        (r"\bCould you please\b\s*", "", re.IGNORECASE),
        (r"\bCan you please\b\s*", "", re.IGNORECASE),
        (r"\bplease\b\s*", "", re.IGNORECASE),
        (r"\bkindly\b\s*", "", re.IGNORECASE),
        (r"\bAs an AI language model,?\s*", "", re.IGNORECASE),
        (r"\bNote that\b\s*", "", re.IGNORECASE),
        (r"\bit is important to note that\b\s*", "", re.IGNORECASE),
        (r"\bin order to\b", "to", re.IGNORECASE),
        (r"\bdue to the fact that\b", "because", re.IGNORECASE),
        (r"\bat this point in time\b", "now", re.IGNORECASE),
        (r"\bfor the purpose of\b", "to", re.IGNORECASE),
        (r"\s{2,}", " "),
    ]
    for pattern, repl, *flags in replacements:
        flag = flags[0] if flags else 0
        result = re.sub(pattern, repl, result, flags=flag)
    return result.strip()


def analyze(prompt: str, model: str = "gpt-3.5-turbo") -> PromptAnalysis:
    tokens = count_tokens(prompt, model)
    redundancy = _redundancy_score(prompt)
    clarity = _clarity_score(prompt)
    suggestions = _build_suggestions(prompt, redundancy, clarity)
    optimized = _apply_optimizations(prompt)
    opt_tokens = count_tokens(optimized, model)
    savings = tokens - opt_tokens

    return PromptAnalysis(
        original_prompt=prompt,
        token_count=tokens,
        char_count=len(prompt),
        redundancy_score=round(redundancy, 3),
        clarity_score=round(clarity, 3),
        suggestions=suggestions,
        optimized_prompt=optimized if savings > 0 else None,
        optimized_token_count=opt_tokens if savings > 0 else None,
        token_savings=savings if savings > 0 else 0,
        savings_pct=round(100 * savings / tokens, 1) if savings > 0 and tokens > 0 else 0.0,
    )


def batch_analyze(prompts: list[str], model: str = "gpt-3.5-turbo") -> list[PromptAnalysis]:
    return [analyze(p, model) for p in prompts]


def report(analysis: PromptAnalysis) -> str:
    lines = [
        "=" * 60,
        "PROMPT ANALYSIS REPORT",
        "=" * 60,
        f"Tokens         : {analysis.token_count}",
        f"Characters     : {analysis.char_count}",
        f"Redundancy     : {analysis.redundancy_score:.2f} {'⚠ High' if analysis.redundancy_score > 0.3 else '✓ OK'}",
        f"Clarity        : {analysis.clarity_score:.2f} {'⚠ Low' if analysis.clarity_score < 0.6 else '✓ OK'}",
        "",
        "SUGGESTIONS:",
    ]
    for i, s in enumerate(analysis.suggestions, 1):
        lines.append(f"  {i}. {s}")

    if analysis.optimized_prompt:
        lines += [
            "",
            "OPTIMIZED PROMPT:",
            f"  {analysis.optimized_prompt}",
            "",
            f"Token savings  : {analysis.token_savings} tokens ({analysis.savings_pct}% reduction)",
        ]
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    sample_prompts = [
        "I would like you to please kindly summarize the following text in order to provide me with a brief overview. Note that it is important to note that you should keep it concise.",
        "Summarize the following text in 3 bullet points. Be concise.",
        "Can you please help me to write a Python function that sorts a list of integers?",
    ]

    print("Running batch analysis...\n")
    results = batch_analyze(sample_prompts)
    for r in results:
        print(report(r))
        print()
