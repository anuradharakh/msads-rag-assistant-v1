from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def normalize_url(url: str) -> str:
    """NORMALIZE URL BY REMOVING FRAGMENTS AND TRAILING SLASH. **"""
    return url.split("#")[0].rstrip("/")


def extract_links(
    html: str,
    base_url: str,
    allowed_domain: str,
    allowed_path_prefix: str = "",
) -> list[str]:
    """EXTRACT INTERNAL LINKS WITH OPTIONAL PATH PREFIX FILTER. **"""

    soup = BeautifulSoup(html, "lxml")
    links = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        if not href:
            continue

        absolute_url = urljoin(base_url, href)
        absolute_url = normalize_url(absolute_url)

        parsed = urlparse(absolute_url)

        if allowed_domain not in parsed.netloc:
            continue

        if allowed_path_prefix and not parsed.path.startswith(allowed_path_prefix):
            continue

        links.add(absolute_url)

    return sorted(links)