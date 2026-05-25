from collections import deque
from typing import Dict, List, Set

import requests

from src.crawler.html_cleaner import clean_html, extract_page_text
from src.crawler.link_extractor import extract_links
from src.utils.logger import log_info, log_warning


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_page(url: str) -> str:
    """FETCH HTML PAGE CONTENT. **"""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    return response.text


def crawl_website(
    root_url: str,
    allowed_domain: str,
    max_pages: int = 50,
) -> List[Dict]:
    """CRAWL WEBSITE AND EXTRACT CLEAN TEXT CONTENT. **"""

    queue = deque([root_url.rstrip("/")])

    visited: Set[str] = set()

    crawled_pages = []

    while queue and len(visited) < max_pages:
        current_url = queue.popleft()

        if current_url in visited:
            continue

        try:
            log_info(f"Crawling: {current_url}")

            html = fetch_page(current_url)

            soup = clean_html(html)

            page_text = extract_page_text(soup)

            page_title = soup.title.string.strip() if soup.title else current_url

            crawled_pages.append(
                {
                    "url": current_url,
                    "title": page_title,
                    "text": page_text,
                }
            )

            visited.add(current_url)

            links = extract_links(
                html=html,
                base_url=current_url,
                allowed_domain=allowed_domain,
                allowed_path_prefix="/education/masters-programs/ms-in-applied-data-science",
            )

            for link in links:
                if link not in visited:
                    queue.append(link)

        except Exception as error:
            log_warning(f"Failed to crawl {current_url}: {error}")

    return crawled_pages