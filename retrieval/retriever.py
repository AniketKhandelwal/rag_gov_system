# retrieval/retriever.py

from embeddings.embedder import get_collection
from config.config import CONFIG
from embeddings.embedder import get_vectorstore

def retrieve_documents(query: str):
    """
    Retrieve top relevant documents from ChromaDB
    """

    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=CONFIG["TOP_K"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # --- SMART FILTERING ---
    # prioritize same source (very important)
    filtered_docs = []
    filtered_meta = []

    if len(metadatas) > 0:
        main_source = metadatas[0]["source"]

        for doc, meta in zip(documents, metadatas):
            if meta["source"] == main_source:
                filtered_docs.append(doc)
                filtered_meta.append(meta)

    # fallback if filtering too aggressive
    if len(filtered_docs) >= 2:
        return filtered_docs, filtered_meta
    else:
        return documents, metadatas
    
def get_langchain_retriever():
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": CONFIG["TOP_K"]}
    )

    return retriever