# config/config.py

import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # Paths
    "CHROMA_PATH": "./chromadb",
    "COLLECTION_NAME": "gov_schemes_rag",
    
    # Data
    "DATA_PATH": "data/raw",
    
    # Chunking
    "CHUNK_SIZE": 400,
    "CHUNK_OVERLAP": 50,
    
    # Embeddings
    "EMBEDDING_MODEL": "BAAI/bge-small-en",
    
    # Retrieval
    "TOP_K": 5,
    
    # Gemini
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GEMINI_MODEL": "models/gemini-2.5-flash"
}