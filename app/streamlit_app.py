"""Chat front end for the API.

    streamlit run app/streamlit_app.py

Set RAG_API_URL if the API is not on localhost:8000.
"""
import os

import requests
import streamlit as st

API = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Equipment manual assistant", page_icon="🔧")
st.title("Equipment manual assistant")
st.caption("Answers come only from the indexed manuals and cite the page they used.")

with st.sidebar:
    k = st.slider("Passages retrieved", 1, 8, 4)
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        st.success(f"API up: {h['chunks']} chunks, {h['embedder']}, {h['llm_model']}")
        if not h.get("llm_reachable"):
            st.warning("The language model is not reachable; answers will fail.")
        m = requests.get(f"{API}/metrics", timeout=5).json()
        if m.get("latency"):
            st.metric("Median latency", f"{m['latency']['p50_ms'] / 1000:.1f} s")
            st.metric("95th percentile", f"{m['latency']['p95_ms'] / 1000:.1f} s")
    except requests.RequestException:
        st.error(f"Cannot reach the API at {API}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["declined"]:
            st.info("No supporting passage was found, so the assistant declined to answer.")
        with st.expander(f"Sources ({len(turn['sources'])}) and timings"):
            for s in turn["sources"]:
                st.markdown(f"**{s['manual']}, p.{s['page']}** (similarity {s['score']:.2f})")
                st.caption(s["snippet"])
            st.json(turn["timings_ms"])

question = st.chat_input("Ask about the manuals")
if question:
    with st.spinner("Searching the manuals..."):
        try:
            r = requests.post(f"{API}/ask", json={"question": question, "k": k}, timeout=300)
            r.raise_for_status()
            st.session_state.history.append(r.json())
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
    st.rerun()
