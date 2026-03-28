# 💬 Government Schemes Conversational RAG System

A production-ready conversational AI system that answers queries about Indian government schemes using Retrieval-Augmented Generation (RAG) with LangChain.

---

## 🚀 Overview

This project implements an end-to-end **Conversational RAG pipeline** that allows users to:

- Upload government scheme documents
- Ask natural language questions
- Get **context-aware answers**
- View **source-backed responses**

The system uses **semantic search + LLM generation** to provide accurate and reliable information.

---

## 🔥 Key Features

- 💬 Conversational AI with memory (multi-turn queries)
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware responses using LangChain
- 📚 Source-backed answers (reduces hallucination)
- 📂 Upload custom `.txt` documents
- ⚡ Real-time interaction via Streamlit UI

---

## 🧠 Tech Stack

- **Python**
- **LangChain**
- **Google Gemini API**
- **ChromaDB (Vector Database)**
- **HuggingFace Embeddings**
- **Streamlit**

---

## ⚙️ System Architecture

```

User Query
↓
Retriever (ChromaDB)
↓
LangChain Conversational Chain (Memory)
↓
Gemini LLM
↓
Answer + Sources

```

---

## 📂 Project Structure

```

rag_gov_system/
│
├── api/                # (FastAPI backend)
├── app/                # Streamlit UI
│   └── streamlit_app.py
│
├── config/             # Configurations (API keys, constants)
├── ingestion/          # Document loading
├── processing/         # Text splitting
├── embeddings/         # Embedding + vector storage
├── retrieval/          # Retriever logic
├── generation/         # LLM + LangChain chains
│
├── data/
│   ├── raw/            # Input documents
│   └── processed/
│
├── chromadb/           # Vector DB storage
├── notebook/           # Experimental notebooks
│
├── requirements.txt
├── lib_install.sh      # Quick setup script
└── README.md

```

---

## 🛠️ Installation

### 🔹 Setup:

```bash
git clone <your-repo-link>
cd rag_gov_system

python -m venv .venv
source .venv/bin/activate   

./lib_install.sh
```

---

## 🔑 Environment Setup

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Application

```bash
python -m streamlit run app/streamlit_app.py
```

Then open:

👉 [http://localhost:8501](http://localhost:8501)

---

## 📌 How to Use

### 1️⃣ Upload Documents

* Upload `.txt` files containing government schemes

### 2️⃣ Process Data

* Click **"Process Files"**
* System builds embeddings and vector database

### 3️⃣ Ask Questions

Example:

* What is Vigyan Dhara?
* Is there a scheme for authors?
* What are its benefits?

### 4️⃣ View Results

* Answer generated using LLM
* Sources shown for transparency

---

## 💬 Example Conversation

```
User: Is there a scheme for young authors?
Assistant: Yes, the YUVA scheme...

User: What are its benefits?
Assistant: The scheme provides...
```

---

## 🧠 Key Concepts Implemented

* Retrieval-Augmented Generation (RAG)
* Semantic Search using Embeddings
* Conversational Memory (LangChain)
* Context-aware Query Handling
* Source-grounded Response Generation

---

## ⚠️ Known Limitations

* Depends on Gemini API quota
* Works best with structured text data
* Currently supports `.txt` files only

---

## 🚀 Future Improvements

* Multi-language support
* PDF ingestion pipeline
* RAGAS evaluation
* Cloud deployment (AWS / Render)
* Advanced query rewriting

---

## 📌 Highlights

* Built full RAG pipeline from scratch
* Upgraded to LangChain conversational system
* Implemented memory + retrieval + generation
* Designed real-world use case (public information access)

---

## 👨‍💻 Author

**Aniket Khandelwal**
AI & Data Science Student
Jaipur, India

* LinkedIn: [https://www.linkedin.com/in/aniket-khandelwal-97b036291/](https://www.linkedin.com/in/aniket-khandelwal-97b036291/)
* GitHub: [https://github.com/AniketKhandelwal](https://github.com/AniketKhandelwal)

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!

---
