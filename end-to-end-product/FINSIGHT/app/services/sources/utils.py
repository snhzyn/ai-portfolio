from __future__ import annotations

import html

import httpx
from bs4 import BeautifulSoup


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)

    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def clean_text(text: str) -> str:
    text = html.unescape(text)
    return " ".join(text.split()).strip()


def extract_title(html_text: str) -> str | None:
    soup = BeautifulSoup(html_text, "html.parser")

    if soup.title and soup.title.string:
        return clean_text(soup.title.string)

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return clean_text(og_title["content"])

    return None