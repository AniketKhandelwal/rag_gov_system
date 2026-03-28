# generation/langchain_generator.py

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from retrieval.retriever import get_langchain_retriever
from config.config import CONFIG
import asyncio


# Initialize LLM
def get_llm():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    llm = ChatGoogleGenerativeAI(
        model=CONFIG["GEMINI_MODEL"],
        google_api_key=CONFIG["GEMINI_API_KEY"],
        temperature=0.3
    )
    return llm


# Initialize memory
def get_memory():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return memory


# Build conversational chain
def get_conversational_chain():
    llm = get_llm()
    retriever = get_langchain_retriever()
    memory = get_memory()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    return chain