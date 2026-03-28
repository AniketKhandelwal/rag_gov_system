# embeddings/embedder.py

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config.config import CONFIG


# Initialize Chroma client
def get_chroma_client():
    return chromadb.PersistentClient(path=CONFIG["CHROMA_PATH"])


# Get or create collection
def get_collection():
    client = get_chroma_client()
    
    collection = client.get_or_create_collection(
        name=CONFIG["COLLECTION_NAME"]
    )
    
    return collection


# Load embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"]
    )
    
    return embedding_model


# Store documents in Chroma
def store_documents(split_docs):
    collection = get_collection()

    documents_text = [doc.page_content for doc in split_docs]
    metadatas = [doc.metadata for doc in split_docs]
    ids = [str(i) for i in range(len(split_docs))]

    collection.add(
        documents=documents_text,
        metadatas=metadatas,
        ids=ids
    )

    return len(documents_text)

def reset_collection():
    client = get_chroma_client()
    
    try:
        client.delete_collection(CONFIG["COLLECTION_NAME"])
    except Exception:
        # Collection may not exist — ignore
        pass

def get_vectorstore():
    """
    Returns LangChain-compatible Chroma vector store
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL"]
    )

    vectorstore = Chroma(
        persist_directory=CONFIG["CHROMA_PATH"],
        embedding_function=embedding_model,
        collection_name=CONFIG["COLLECTION_NAME"]
    )

    return vectorstore