import streamlit as st
import json
from optimizer import analyze, report, count_tokens

st.set_page_config(page_title="LLM Prompt Optimizer", layout="wide")

st.title("LLM Prompt Optimizer")
st.markdown("Analyze your prompts for **token efficiency**, redundancy, and clarity — then get an optimized version.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Prompt")
    model = st.selectbox("Tokenizer model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"], index=0)
    prompt = st.text_area("Paste your prompt here", height=250, placeholder="e.g. I would like you to please summarize the following text...")
    analyze_btn = st.button("Analyze & Optimize", type="primary")

with col2:
    st.subheader("Live Token Counter")
    if prompt:
        live_count = count_tokens(prompt, model)
        st.metric("Current token count", live_count)
        st.caption(f"Estimated cost @ $0.002/1K tokens: ${live_count * 0.002 / 1000:.6f}")

if analyze_btn and prompt.strip():
    result = analyze(prompt, model)

    st.divider()
    st.subheader("Analysis Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tokens", result.token_count)
    m2.metric("Token Savings", f"-{result.token_savings}" if result.token_savings else "0")
    m3.metric("Redundancy", f"{result.redundancy_score:.2f}", delta="high" if result.redundancy_score > 0.3 else None, delta_color="inverse")
    m4.metric("Clarity", f"{result.clarity_score:.2f}", delta="low" if result.clarity_score < 0.6 else None, delta_color="normal")

    st.subheader("Suggestions")
    for s in result.suggestions:
        st.markdown(f"- {s}")

    if result.optimized_prompt:
        st.subheader("Optimized Prompt")
        st.success(result.optimized_prompt)
        st.caption(f"Reduced from **{result.token_count}** to **{result.optimized_token_count}** tokens — saved **{result.savings_pct}%**")

        if st.button("Copy optimized prompt"):
            st.toast("Copied to clipboard!")

    with st.expander("Raw JSON output"):
        import dataclasses
        st.json(dataclasses.asdict(result))

elif analyze_btn:
    st.warning("Please enter a prompt first.")
