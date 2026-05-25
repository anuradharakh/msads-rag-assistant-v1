import re
import hashlib
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


HEADER_PATTERNS = [
    r"^#{1,6}\s+.+",
    r"^[A-Z][A-Za-z0-9\s\-\&]{3,}$",
]

def make_chunk_id(url: str, section_index: int, chunk_index: int, chunk_text: str) -> str:
    """CREATE UNIQUE STABLE CHUNK ID. **"""
    raw = f"{url}_{section_index}_{chunk_index}_{chunk_text[:80]}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"

def is_header(line: str) -> bool:
    """CHECK IF A LINE LOOKS LIKE A HEADER. **"""

    stripped = line.strip()

    if len(stripped) > 120:
        return False

    for pattern in HEADER_PATTERNS:
        if re.match(pattern, stripped):
            return True

    return False


def split_into_sections(text: str) -> List[Dict]:
    """SPLIT DOCUMENT INTO HEADER-BASED SECTIONS. **"""

    lines = text.splitlines()

    sections = []

    current_header = "Introduction"
    current_content = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if is_header(stripped):
            if current_content:
                sections.append(
                    {
                        "header": current_header,
                        "content": "\n".join(current_content),
                    }
                )

            current_header = stripped
            current_content = []

        else:
            current_content.append(stripped)

    if current_content:
        sections.append(
            {
                "header": current_header,
                "content": "\n".join(current_content),
            }
        )

    return sections


def create_header_aware_chunks(
    documents: List[Dict],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """CREATE HEADER-AWARE RECURSIVE CHUNKS. **"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []

    for document in documents:
        url = document["url"]
        title = document["title"]
        text = document["text"]

        sections = split_into_sections(text)

        for section_index, section in enumerate(sections):
            header = section["header"]

            section_text = (
                f"Title: {title}\n"
                f"Section: {header}\n\n"
                f"{section['content']}"
            )

            section_chunks = splitter.split_text(section_text)

            for chunk_index, chunk_text in enumerate(section_chunks):
                chunk_id = make_chunk_id(
                            url=url,
                            section_index=section_index,
                            chunk_index=chunk_index,
                            chunk_text=chunk_text,
                        )

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "metadata": {
                            "url": url,
                            "title": title,
                            "section": header,
                            "chunk_type": "header_recursive",
                        },
                    }
                )

    return chunks