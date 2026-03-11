from __future__ import annotations
from typing import TypedDict

class SourceConfig(TypedDict):
    name: str
    source_type: str  # official, wire, media, corporate, international_org
    category: str
    base_url: str
    notes: str


SOURCE_REGISTRY: dict[str, dict[str, list[SourceConfig] | int]] = {
    "macro": {
        "max_items": 5,
        "primary": [
            {
                "name": "Federal Reserve FOMC",
                "source_type": "official",
                "category": "monetary_policy",
                "base_url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                "notes": "Use for FOMC calendars, statements, and policy guidance.",
            },
            {
                "name": "ECB Press Releases",
                "source_type": "official",
                "category": "monetary_policy",
                "base_url": "https://www.ecb.europa.eu/press/pr/date/html/index.en.html",
                "notes": "Use for ECB policy decisions, speeches, and official releases.",
            },
            {
                "name": "BOJ Announcements",
                "source_type": "official",
                "category": "monetary_policy",
                "base_url": "https://www.boj.or.jp/en/announcements/release_2026/index.htm/",
                "notes": "Use for BOJ announcements and policy statements.",
            },
            {
                "name": "U.S. BLS CPI Releases",
                "source_type": "official",
                "category": "economic_data",
                "base_url": "https://www.bls.gov/cpi/",
                "notes": "Use for CPI and other major U.S. labor and inflation indicators.",
            },
        ],
        "secondary": [
            {
                "name": "Reuters Markets",
                "source_type": "wire",
                "category": "market_coverage",
                "base_url": "https://www.reuters.com/markets/",
                "notes": "Use as trusted secondary market interpretation for macro developments.",
            }
        ],
    },
    "markets": {
        "max_items": 5,
        "primary": [
            {
                "name": "Reuters Markets",
                "source_type": "wire",
                "category": "equities",
                "base_url": "https://www.reuters.com/markets/",
                "notes": "Use for broad market developments, equity moves, and risk sentiment.",
            },
            {
                "name": "Reuters Global Market Data",
                "source_type": "wire",
                "category": "market_data",
                "base_url": "https://www.reuters.com/markets/global-market-data/",
                "notes": "Use for global market data context and price-action summaries.",
            },
        ],
        "secondary": [
            {
                "name": "Corporate IR / Earnings Releases",
                "source_type": "corporate",
                "category": "earnings",
                "base_url": "",
                "notes": "Use official investor relations pages when Reuters references earnings or guidance.",
            },
            {
                "name": "CNBC",
                "source_type": "media",
                "category": "market_commentary",
                "base_url": "https://www.cnbc.com/markets/",
                "notes": "Optional supporting source only; do not rely on CNBC alone for verification.",
            },
        ],
    },
    "commodities_fx": {
        "max_items": 5,
        "primary": [
            {
                "name": "Reuters Commodities",
                "source_type": "wire",
                "category": "commodities",
                "base_url": "https://www.reuters.com/markets/commodities/",
                "notes": "Use for oil, gas, metals, and commodity market developments.",
            },
            {
                "name": "Reuters Currencies",
                "source_type": "wire",
                "category": "fx",
                "base_url": "https://www.reuters.com/markets/currencies/",
                "notes": "Use for FX developments including USD strength, EM FX, and safe-haven flows.",
            },
            {
                "name": "Reuters Rates & Bonds",
                "source_type": "wire",
                "category": "rates",
                "base_url": "https://www.reuters.com/markets/rates-bonds/",
                "notes": "Use for Treasury yields, bond market moves, and rate-driven asset repricing.",
            },
            {
                "name": "EIA Weekly Petroleum Status Report",
                "source_type": "official",
                "category": "energy_data",
                "base_url": "https://www.eia.gov/petroleum/supply/weekly/",
                "notes": "Use for U.S. oil inventory, supply, and petroleum balance data.",
            },
            {
                "name": "EIA Short-Term Energy Outlook",
                "source_type": "official",
                "category": "energy_outlook",
                "base_url": "https://www.eia.gov/outlooks/steo/",
                "notes": "Use for broader energy market outlook and production/demand expectations.",
            },
        ],
        "secondary": [
            {
                "name": "U.S. Treasury",
                "source_type": "official",
                "category": "rates_data",
                "base_url": "https://home.treasury.gov/",
                "notes": "Use when official Treasury yield or debt-market data is required.",
            }
        ],
    },
    "geopolitical": {
        "max_items": 5,
        "primary": [
            {
                "name": "Reuters World",
                "source_type": "wire",
                "category": "geopolitics",
                "base_url": "https://www.reuters.com/world/",
                "notes": "Use for global conflicts, sanctions, and cross-border political developments.",
            },
            {
                "name": "Reuters Markets Geopolitics Coverage",
                "source_type": "wire",
                "category": "market_geopolitics",
                "base_url": "https://www.reuters.com/markets/",
                "notes": "Use for geopolitical developments with direct market implications.",
            },
            {
                "name": "OFAC Sanctions",
                "source_type": "official",
                "category": "sanctions",
                "base_url": "https://ofac.treasury.gov/",
                "notes": "Use for official U.S. sanctions announcements and updates.",
            },
            {
                "name": "USTR News / Releases",
                "source_type": "official",
                "category": "trade_policy",
                "base_url": "https://ustr.gov/about-us/policy-offices/press-office/press-releases",
                "notes": "Use for official trade actions, tariffs, and trade-related policy announcements.",
            },
        ],
        "secondary": [
            {
                "name": "Official Government / International Organization Sources",
                "source_type": "international_org",
                "category": "official_external",
                "base_url": "",
                "notes": "Use only as needed for confirmation of major geopolitical developments.",
            }
        ],
    },
}