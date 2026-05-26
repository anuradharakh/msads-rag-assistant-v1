import hashlib
import re
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


HEADER_PATTERNS = [
    r"^#{1,6}\s+.+",
    r"^[A-Z][A-Za-z0-9\s\-\&]{3,}$",
]


FOUNDATIONAL_COURSES = [
    "Introduction to Statistical Concepts",
    "R for Data Science",
    "Python for Data Science",
    "Advanced Linear Algebra for Machine Learning",
]

CORE_COURSES = [
    "Statistical Models for Data Science",
    "Leadership and Consulting for Data Science",
    "Data Engineering Platforms for Analytics",
    "Big Data and Cloud Computing",
    "Machine Learning I",
    "Time Series Analysis and Forecasting",
    "Machine Learning II",
]

CAPSTONE_COURSES = [
    "Data Science Capstone Project",
]

SEMINAR_COURSES = [
    "Career Seminar",
]

ELECTIVE_COURSES = [
    "Generative AI Principles",
    "Advanced Computer Vision with Deep Learning",
    "Advanced Machine Learning and Artificial Intelligence",
    "Bayesian Machine Learning with GenAI Applications",
    "Data Science for Algorithmic Marketing",
    "Data Visualization Techniques",
    "Digital Marketing Analytics in Theory and Practice",
    "Financial Analytics",
    "Health Analytics",
    "Machine Learning Operations",
    "Natural Language Processing and Cognitive Computing",
    "Real Time Intelligent Systems",
    "Reinforcement Learning",
    "Supply Chain Optimization",
    "Quantitative Finance",
]


def make_chunk_id(url: str, section_index: int, chunk_index: int, chunk_text: str) -> str:
    """CREATE UNIQUE STABLE CHUNK ID. **"""
    raw = f"{url}_{section_index}_{chunk_index}_{chunk_text[:120]}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


def is_header(line: str) -> bool:
    """CHECK WHETHER LINE LOOKS LIKE A SECTION HEADER. **"""
    stripped = line.strip()

    if len(stripped) > 120:
        return False

    for pattern in HEADER_PATTERNS:
        if re.match(pattern, stripped):
            return True

    return False


def split_into_sections(text: str) -> List[Dict]:
    """SPLIT TEXT INTO SIMPLE HEADER-BASED SECTIONS. **"""
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


def _filter_present_courses(text: str, courses: List[str]) -> List[str]:
    """KEEP ONLY COURSES FOUND IN PAGE TEXT. **"""
    lower_text = text.lower()

    return [
        course
        for course in courses
        if course.lower() in lower_text
    ]


def create_curriculum_catalog_chunk(document: Dict) -> List[Dict]:
    """CREATE ONE AGGREGATED COURSE CATALOG CHUNK FOR COURSE-PROGRESSION PAGE. **"""

    url = document["url"]
    title = document["title"]
    text = document["text"]

    foundational = _filter_present_courses(text, FOUNDATIONAL_COURSES)
    core = _filter_present_courses(text, CORE_COURSES)
    capstone = _filter_present_courses(text, CAPSTONE_COURSES)
    seminar = _filter_present_courses(text, SEMINAR_COURSES)
    electives = _filter_present_courses(text, ELECTIVE_COURSES)

    if not any([foundational, core, capstone, seminar, electives]):
        return []

    catalog_text = f"""
Title: {title}
Section: MSADS Course Catalog

The MS in Applied Data Science curriculum includes the following course categories and course names.

Foundational Courses:
{chr(10).join(f"- {course}" for course in foundational) if foundational else "- Not found in extracted page text"}

Core Courses:
{chr(10).join(f"- {course}" for course in core) if core else "- Not found in extracted page text"}

Capstone:
{chr(10).join(f"- {course}" for course in capstone) if capstone else "- Not found in extracted page text"}

Seminar:
{chr(10).join(f"- {course}" for course in seminar) if seminar else "- Not found in extracted page text"}

Elective Examples:
{chr(10).join(f"- {course}" for course in electives) if electives else "- Not found in extracted page text"}
""".strip()

    return [
        {
            "chunk_id": make_chunk_id(
                url=url,
                section_index=999,
                chunk_index=0,
                chunk_text=catalog_text,
            ),
            "chunk_text": catalog_text,
            "metadata": {
                "url": url,
                "title": title,
                "section": "MSADS Course Catalog",
                "chunk_type": "curriculum_catalog",
            },
        }
    ]


def create_header_aware_chunks(
    documents: List[Dict],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """CREATE HEADER-AWARE CHUNKS WITH SPECIAL CURRICULUM CATALOG CHUNKS. **"""

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

        if "course-progressions" in url:
            chunks.extend(create_curriculum_catalog_chunk(document))

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