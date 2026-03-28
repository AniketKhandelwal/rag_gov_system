# generation/generator.py

import google.generativeai as genai
from config.config import CONFIG
from retrieval.retriever import retrieve_documents

if not CONFIG["GEMINI_API_KEY"]:
    raise ValueError("Missing Gemini API key")

# Initialize Gemini once
def get_llm():
    genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
    model = genai.GenerativeModel(CONFIG["GEMINI_MODEL"])
    return model


# Generate answer
def generate_answer(query: str):
    """
    Full RAG pipeline:
    Query → Retrieve → Generate answer
    """

    try:
        model = get_llm()

        docs, metas = retrieve_documents(query)

        if not docs:
            return "No relevant information found."

        context = "\n\n".join(docs)

        prompt = f"""
You are an expert assistant for Indian government schemes.

Instructions:
- Answer ONLY using the provided context
- If answer is not present, say "I don't know"
- Be clear, structured, and concise
- Do not hallucinate

Context:
{context}

Question:
{query}

Answer:
"""

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"[Generation Error]: {str(e)}"