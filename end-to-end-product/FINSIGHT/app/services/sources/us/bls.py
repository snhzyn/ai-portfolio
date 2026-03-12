import html
import httpx
from bs4 import BeautifulSoup


def _parse_bls_cpi(
    source: dict[str, Any],
    url: str,
    html_text: str,
    target_date: date,
) -> RawSourceItem | None:
    title = _extract_title(html_text) or "U.S. BLS CPI Update"
    snippet = _select_bls_snippet(html_text)

    print(f"[PARSE][BLS] title={title}")
    print(f"[PARSE][BLS] snippet={snippet[:200] if snippet else 'NONE'}")

    return RawSourceItem(
        source_name=source["name"],
        source_type=source["source_type"],
        category=source["category"],
        url=url,
        headline=title,
        publish_date=target_date,
        content=snippet or "BLS CPI page fetched successfully.",
    )


def _select_bls_snippet(html_text: str, max_length: int = 300) -> str:
    """
    Select snippet prioritizing CPI release language.
    """
    soup = BeautifulSoup(html_text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = _extract_container_paragraphs(soup)

    prioritized = _best_matching_paragraph(
        paragraphs,
        keywords=[
            "consumer price index for all urban consumers",
            "rose",
            "increased",
            "seasonally adjusted",
            "over the last 12 months",
            "cpi-u",
            "news release",
            "index level",
            "all items index",
            "monthly",
        ],
        boilerplate_check=_is_bls_boilerplate,
    )
    if prioritized:
        return _truncate(prioritized, max_length)

    for paragraph in paragraphs:
        lower = paragraph.lower()
        if _is_bls_boilerplate(lower):
            continue
        if len(paragraph) >= 80:
            return _truncate(paragraph, max_length)

    return _extract_snippet(html_text, max_length=max_length)


def _is_bls_boilerplate(text: str) -> bool:
    """
    Identify common boilerplate text on BLS / .gov pages.
    """
    boilerplate_phrases = [
        "the .gov means it's official",
        "the .gov means its official",
        "federal government websites often end in .gov",
        "before sharing sensitive information",
        "make sure you're on a federal government site",
        "make sure youre on a federal government site",
        "an official website of the united states government",
        "here is how you know",
        "here's how you know",
        "the site is secure",
        "the https:// ensures",
        "that any information you provide is encrypted",
        "official government organization in the united states",    
        "secure .gov websites use https",
        "a lock",
        "locked padlock icon",
        "means you've safely connected to the .gov website",
        "share sensitive information only on official, secure websites",
        "official, secure websites",
    ]

    return any(phrase in text for phrase in boilerplate_phrases)