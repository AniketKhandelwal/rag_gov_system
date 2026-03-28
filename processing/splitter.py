# processing/splitter.py

import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CONFIG


def split_documents(documents):
    """
    Hybrid splitter:
    1. Structured splitting using scheme sections
    2. Fallback to semantic chunking
    """

    split_docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["CHUNK_SIZE"],
        chunk_overlap=CONFIG["CHUNK_OVERLAP"]
    )

    for doc in documents:
        text = doc.page_content

        # Structured splitting
        sections = re.split(
            r"(Scheme Name:|Description:|Objective:|Eligibility:|Benefits:|Application:|Components:)",
            text
        )

        # If structured sections exist
        if len(sections) > 3:
            for i in range(1, len(sections), 2):
                try:
                    section_title = sections[i].strip()
                    section_content = sections[i + 1].strip()

                    if len(section_content) < 20:
                        continue

                    chunk_text = f"{section_title}\n{section_content}"

                    split_docs.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "section": section_title
                            }
                        )
                    )

                except Exception:
                    continue

        else:
            # Fallback semantic splitting
            chunks = splitter.split_text(text)

            for chunk in chunks:
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "section": "general"
                        }
                    )
                )

    return split_docs