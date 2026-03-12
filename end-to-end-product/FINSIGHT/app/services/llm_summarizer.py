import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MAX_INPUT_CHARS = 2000
ENABLE_LLM_SUMMARY = os.getenv("ENABLE_LLM_SUMMARY", "false").lower() == "true"

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def summarize_for_finance_briefing(
    text: str,
    source_name: str,
    headline: str,
) -> str:
    print("ENABLE_LLM_SUMMARY =", ENABLE_LLM_SUMMARY)
    print("ANTHROPIC_API_KEY exists =", bool(os.getenv("ANTHROPIC_API_KEY")))
    print("LLM summarizing:", source_name)

    if not ENABLE_LLM_SUMMARY:
        print("LLM disabled, returning raw text")
        return text[:300]

    if not text:
        return ""

    truncated = text[:MAX_INPUT_CHARS]

    prompt = f"""
You are a financial intelligence analyst.

Summarize the following source text in 1-2 plain sentences for a daily macro/market intelligence briefing.

Rules:
- Return plain text only.
- Do not use headings, markdown, bullets, labels, or prefixes.
- If the text contains no current market-relevant information, return exactly:
NO_MARKET_SIGNAL
- Focus only on monetary policy, inflation, economic growth, financial conditions, or market implications.

Source: {source_name}
Headline: {headline}

Text:
{truncated}
"""

    print("Calling Anthropic API...")
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    print("Anthropic API call completed")

    summary = response.content[0].text.strip()

    # LLM formatting cleanup
    summary = summary.replace("# Summary", "")
    summary = summary.replace("\n", " ")
    summary = summary.strip()

    return summary