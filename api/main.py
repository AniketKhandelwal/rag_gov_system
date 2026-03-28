# api/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
import shutil
import logging

from ingestion.loader import load_txt_files
from processing.splitter import split_documents
from embeddings.embedder import store_documents, reset_collection
from generation.generator import generate_answer


# ------------------- CONFIG -------------------

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------- APP INIT -------------------

app = FastAPI(
    title="Government Schemes RAG API",
    version="1.0.0",
    description="Production-ready RAG system for government scheme Q&A"
)

# ------------------- ROOT ENDPOINT ------------------- 

@app.get("/")
def root():
    return {"message": "Welcome to the Government Schemes RAG API. Use /docs for API documentation."}


# ------------------- HEALTH CHECK -------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ------------------- FILE UPLOAD -------------------

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple TXT files and build vector database
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved_files = []

    try:
        for file in files:

            if not file.filename.endswith(".txt"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}"
                )

            file_path = os.path.join(UPLOAD_DIR, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(file.filename)

        logger.info(f"Saved files: {saved_files}")

        # 🔥 Reset DB before re-ingestion (optional but recommended)
        reset_collection()

        # Run pipeline
        docs = load_txt_files(UPLOAD_DIR)

        if not docs:
            raise HTTPException(status_code=500, detail="No valid documents found")

        split_docs = split_documents(docs)

        if not split_docs:
            raise HTTPException(status_code=500, detail="Splitting failed")

        count = store_documents(split_docs)

        return {
            "status": "success",
            "files_uploaded": saved_files,
            "documents_loaded": len(docs),
            "chunks_created": len(split_docs),
            "chunks_stored": count
        }

    except Exception as e:
        logger.error(f"Upload pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------- QUERY ENDPOINT -------------------

@app.get("/ask")
def ask(query: str):
    """
    Query the RAG system
    """

    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Invalid query")

    try:
        answer = generate_answer(query)

        return {
            "status": "success",
            "query": query,
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Error generating response"
        )
    
# ------------------- RESET ENDPOINT -------------------

@app.post("/reset")
def reset_db():
    reset_collection()
    return {"status": "database reset"}