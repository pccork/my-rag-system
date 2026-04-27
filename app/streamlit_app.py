from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_system.config import get_settings
from rag_system.query import format_citation_list, query
from scripts.query import parse_filters


st.set_page_config(page_title="Local RAG", layout="wide")

settings = get_settings()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

st.title("Local RAG")
st.caption("Ask questions against your indexed PDFs.")

with st.sidebar:
    st.header("Audit")
    user_id = st.text_input("User ID", value="streamlit-user")
    st.caption(f"Session: {st.session_state.session_id}")

    st.header("Retrieval")
    top_k = st.slider(
        "Top-k chunks",
        min_value=1,
        max_value=12,
        value=settings.retrieval_top_k,
    )
    show_debug = st.checkbox("Show retrieved chunks", value=False)

    st.header("Filters")
    document_type = st.selectbox("Document type", ["Any", "SOP", "IFU"])
    maintenance_only = st.checkbox("Maintenance only")
    warning_only = st.checkbox("Warnings only")

question = st.text_input("Question", placeholder="What does the procedure say about cleaning?")
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    filter_values: list[str] = []
    if document_type != "Any":
        filter_values.append(f"document_type={document_type}")
    if maintenance_only:
        filter_values.append("is_maintenance=true")
    if warning_only:
        filter_values.append("contains_warning=true")

    try:
        with st.spinner("Searching documents and generating an answer..."):
            response = query(
                question.strip(),
                top_k=top_k,
                filters=parse_filters(filter_values),
                user_id=user_id.strip() or "streamlit-user",
                session_id=st.session_state.session_id,
                request_metadata={"channel": "streamlit"},
            )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.subheader("Answer")
        st.write(response.answer)
        st.caption(f"Audit ID: {response.audit_id}")

        st.subheader("Cited Sources")
        if response.citations:
            for citation in response.citations:
                score = f"{citation.score:.3f}" if citation.score is not None else "n/a"
                st.markdown(
                    f"[{citation.index}] **{citation.filename}** "
                    f"- page {citation.page} - section: {citation.section} "
                    f"- version: {citation.version} "
                    f"- score: {score}"
                )
        else:
            st.info("No cited sources returned.")

        if show_debug:
            st.subheader("Retrieved Chunks")
            for index, result in enumerate(response.results, start=1):
                citation = response.citations[index - 1] if index <= len(response.citations) else None
                title = format_citation_list([citation]) if citation else f"Chunk {index}"
                with st.expander(title):
                    st.write(result.text)
                    st.json(result.metadata)
