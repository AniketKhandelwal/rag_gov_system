# ingestion/loader.py

import os
from langchain_core.documents import Document
from config.config import CONFIG


def load_txt_files(folder_path: str = None):
    """
    Load all .txt files from a folder and convert them into LangChain Documents.
    """

    if folder_path is None:
        folder_path = CONFIG["DATA_PATH"]

    documents = []

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                text = text.strip()

                # Skip very small content
                if len(text) < 50:
                    continue

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file,
                        "type": "government_scheme",
                        "category": "general"
                    }
                )

                documents.append(doc)

            except Exception as e:
                print(f"[Loader Error] {file}: {e}")

    return documents