import streamlit as st
from core.retriever import get_rag_chain
import os


st.set_page_config(page_title="basicLLM — RAG Chat", layout="wide")

st.title("basicLLM — RAG Chat")

with st.sidebar:
    k = st.slider("Top-k results", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
    st.markdown("---")
    st.markdown("Provide your OpenAI API key in `core/.env` or set `OPENAI_API_KEY` environment variable.")


if "history" not in st.session_state:
    st.session_state.history = []


def run_query(query: str):
    chain = get_rag_chain(k=k, temperature=temperature)
    # chain returns dict when return_source_documents=True
    res = chain({"query": query})
    return res


with st.form("query_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question about your PDFs:")
    submit = st.form_submit_button("Send")

if submit and user_input:
    with st.spinner("Running RAG retrieval..."):
        try:
            out = run_query(user_input)
        except Exception as e:
            st.error(f"Error running query: {e}")
            out = None

    if out:
        answer = out.get("result") or out.get("answer") or out.get("response") or str(out)
        docs = out.get("source_documents") or []

        st.session_state.history.append({"query": user_input, "answer": answer, "docs": docs})

for entry in reversed(st.session_state.history):
    st.markdown("---")
    st.markdown(f"**Q:** {entry['query']}")
    st.markdown(f"**A:** {entry['answer']}")
    if entry["docs"]:
        st.markdown("**Sources:**")
        for d in entry["docs"]:
            src = getattr(d, "metadata", {}).get("source") if hasattr(d, "metadata") else None
            st.markdown(f"- {src}")
