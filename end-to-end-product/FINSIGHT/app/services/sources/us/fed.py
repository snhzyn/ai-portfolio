from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from bs4 import BeautifulSoup

from app.services.source_models import RawSourceItem
from app.services.sources.utils import fetch_html, clean_text


FED_PRESS_BASE_URL = "https://www.federalreserve.gov/newsevents/pressreleases"
FED_PRESS_CATEGORIES = ("monetary",)
FED_SUFFIXES = ("a", "b", "c", "d")
FED_LOOKBACK_DAYS = 45


def fetch_fed_press_release_item(source: dict[str, Any], target_date: date) -> RawSourceItem | None:
    release = get_latest_fed_press_release(target_date)
    print(f"[FED] selected_release={release}")

    if not release:
        print("[FED] no release found")
        return None

    html_text = release["html_text"]
    parsed = parse_fed_press_release(html_text)
    print(f"[FED] parsed is None = {parsed is None}")

    if not parsed:
        return None

    publish_date = parsed["publish_date"] or release["date"] or target_date

    content_parts: list[str] = []
    if parsed["release_time"]:
        content_parts.append(f"Release time: {parsed['release_time']}")
    if parsed["content"]:
        content_parts.append(parsed["content"])

    content = "\n\n".join(content_parts).strip()

    if not content:
        print("[FED] content empty")
        return None

    print(f"[FED] headline={parsed.get('headline')}")
    print(f"[FED] publish_date={publish_date}")
    print(f"[FED] release_time={parsed.get('release_time')}")
    print(f"[FED] content_preview={content[:200]}")

    return RawSourceItem(
        source_name=source["name"],
        source_type=source["source_type"],
        category=source["category"],
        url=release["url"],
        headline=parsed["headline"] or "Federal Reserve Press Release",
        publish_date=publish_date,
        content=content,
    )


def get_latest_fed_press_release(target_date: date) -> dict[str, Any] | None:
    """
    Search backward from target_date across Fed press release categories.

    Strategy:
    - For each day, try categoryYYYYMMDD[a-d].htm
    - Collect all existing hits on that day
    - Select the best hit for that day using mild category/title priority
    - Return the first day that has any valid hit
    """
    for offset in range(FED_LOOKBACK_DAYS + 1):
        candidate_date = target_date - timedelta(days=offset)
        ymd = candidate_date.strftime("%Y%m%d")

        daily_hits: list[dict[str, Any]] = []

        for category in FED_PRESS_CATEGORIES:
            for suffix in FED_SUFFIXES:
                url = f"{FED_PRESS_BASE_URL}/{category}{ymd}{suffix}.htm"

                try:
                    html_text = fetch_html(url)
                except Exception:
                    continue

                title = _extract_page_title(html_text)
                score = _fed_press_priority_score(category, title, url)

                hit = {
                    "url": url,
                    "category": category,
                    "title": title,
                    "score": score,
                    "date": candidate_date,
                    "html_text": html_text,
                }
                daily_hits.append(hit)

                print(
                    f"[FED] hit date={candidate_date} | category={category} | "
                    f"url={url} | title={title} | score={score}"
                )

        if not daily_hits:
            continue

        daily_hits.sort(key=lambda item: item["score"], reverse=True)
        selected = daily_hits[0]
        print(f"[FED] selected_for_day={selected['url']} | score={selected['score']}")
        return selected

    return None


def parse_fed_press_release(html_text: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html_text, "html.parser")

    article = soup.find("div", id="article")
    print(f"[FED] article found = {article is not None}")
    if not article:
        return None

    title_tag = article.select_one("h3.title")
    date_tag = article.select_one("p.article_time")
    release_time_tag = article.select_one("p.releaseTime")

    headline = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else None

    publish_date = None
    if date_tag:
        date_text = clean_text(date_tag.get_text(" ", strip=True))
        publish_date = _parse_fed_date(date_text)

    release_time = clean_text(release_time_tag.get_text(" ", strip=True)) if release_time_tag else None
    if release_time:
        release_time = re.sub(r"\s+Share$", "", release_time).strip()

    body_paragraphs: list[str] = []

    for p in article.find_all("p"):
        classes = p.get("class", [])
        if "article_time" in classes or "releaseTime" in classes:
            continue

        text = clean_text(p.get_text(" ", strip=True))
        if not text:
            continue
        if _is_fed_body_boilerplate(text):
            continue

        body_paragraphs.append(text)

    content = "\n\n".join(body_paragraphs).strip()

    if publish_date:
        date_str = publish_date.strftime("%B %d, %Y")
        if content.startswith(date_str):
            content = content[len(date_str):].strip()

    if not publish_date:
        text_blob = clean_text(article.get_text(" ", strip=True))
        m = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}",
            text_blob,
        )
        if m:
            publish_date = _parse_fed_date(m.group(0))

    if not headline and not content:
        print("[FED] no headline and no content")
        return None

    return {
        "headline": headline,
        "publish_date": publish_date,
        "release_time": release_time,
        "content": content,
    }


def _extract_page_title(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    article = soup.find("div", id="article")
    if article:
        title_tag = article.select_one("h3.title")
        if title_tag:
            return clean_text(title_tag.get_text(" ", strip=True))

    if soup.title and soup.title.string:
        return clean_text(soup.title.string)

    return ""


def _fed_press_priority_score(category: str, title: str, url: str) -> int:
    """
    Mild ranking only.
    Recent day selection happens first; within the same day we weakly prefer:
    monetary > bcreg > other > enforcement > orders
    and statement/minutes over lower-signal docs.
    """
    title_lower = title.lower()
    score = 0

    category_bonus = {
        "monetary": 20,
        "bcreg": 10,
        "other": 8,
        "enforcement": 6,
        "orders": 4,
    }
    score += category_bonus.get(category, 0)

    if "issues fomc statement" in title_lower:
        score += 30
    elif "statement" in title_lower:
        score += 18

    if "minutes" in title_lower:
        score += 12

    if "federal open market committee" in title_lower:
        score += 12
    if "fomc" in title_lower:
        score += 12

    # 완전 배제는 하지 않고 약하게만 조정
    if "discount rate" in title_lower:
        score -= 6
    if "board" in title_lower and "minutes" in title_lower:
        score -= 4
    if "longer-run goals" in title_lower or "strategy" in title_lower or "reaffirms" in title_lower:
        score -= 3

    # 같은 날짜/유형이면 a를 약하게 우선
    if url.endswith("a.htm"):
        score += 2

    return score


def _parse_fed_date(text: str) -> date | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _is_fed_body_boilerplate(text: str) -> bool:
    lower = text.lower()
    boilerplate_phrases = [
        "for media inquiries",
        "implementation note",
        "last update",
        "back to top",
        "return to text",
        "share",
    ]
    return any(phrase in lower for phrase in boilerplate_phrases)