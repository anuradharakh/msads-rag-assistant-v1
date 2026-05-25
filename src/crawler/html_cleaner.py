from bs4 import BeautifulSoup


REMOVE_TAGS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "form",
    "noscript",
    "svg",
]


def clean_html(html: str) -> BeautifulSoup:
    """CLEAN HTML CONTENT FOR TEXT EXTRACTION. **"""

    soup = BeautifulSoup(html, "lxml")

    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    return soup


def extract_page_text(soup: BeautifulSoup) -> str:
    """EXTRACT CLEAN PAGE TEXT. **"""

    text = soup.get_text(separator="\n")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)