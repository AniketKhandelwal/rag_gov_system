# app/streamlit_app.py

import streamlit as st
import os

from ingestion.loader import load_txt_files
from processing.splitter import split_documents
from embeddings.embedder import store_documents, reset_collection
from generation.langchain_generator import get_conversational_chain


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Gov Schemes RAG", layout="wide")

st.title("💬 Government Schemes Conversational RAG")

st.markdown("""
### 🤖 Features:
- Conversational AI for government schemes
- Context-aware answers using LangChain RAG
- Source-backed responses for transparency
""")

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------- SESSION STATE ----------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []


# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt files",
    type=["txt"],
    accept_multiple_files=True
)

if st.sidebar.button("🚀 Process Files"):
    if uploaded_files:

        # Save uploaded files
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        with st.spinner("⚙️ Processing pipeline..."):

            # Reset DB safely
            try:
                reset_collection()
            except:
                pass

            # Run pipeline
            docs = load_txt_files(UPLOAD_DIR)
            split_docs = split_documents(docs)
            store_documents(split_docs)

            # Initialize LangChain conversational chain
            st.session_state.qa_chain = get_conversational_chain()

        st.success("✅ System ready! Start chatting.")

    else:
        st.warning("⚠️ Please upload files first.")


# ---------------- CONTROLS ----------------
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history_ui = []
    st.session_state.qa_chain = get_conversational_chain()
    st.success("Chat reset")

if st.sidebar.button("🗑 Reset Database"):
    try:
        reset_collection()
        st.session_state.qa_chain = None
        st.session_state.chat_history_ui = []
        st.success("Database reset")
    except:
        st.warning("No database found")


# ---------------- CHAT UI ----------------
st.header("💬 Chat")

# Display chat history
for role, message in st.session_state.chat_history_ui:
    with st.chat_message(role):
        st.markdown(message)


# Chat input
user_input = st.chat_input("Ask your question about government schemes...")

if user_input:

    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload and process documents first.")
        st.stop()

    # Display user message
    st.chat_message("user").markdown(user_input)

    with st.spinner("🤔 Thinking..."):
        response = st.session_state.qa_chain.invoke({
            "question": user_input
        })

    answer = response.get("answer", "No answer generated.")
    sources = response.get("source_documents", [])

    # Display assistant answer
    with st.chat_message("assistant"):
        st.markdown(answer)

        # 🔥 Show sources
        if sources:
            with st.expander("📚 Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'unknown')}")
                    st.markdown(doc.page_content[:300] + "...")


    # Save chat history
    st.session_state.chat_history_ui.append(("user", user_input))
    st.session_state.chat_history_ui.append(("assistant", answer))