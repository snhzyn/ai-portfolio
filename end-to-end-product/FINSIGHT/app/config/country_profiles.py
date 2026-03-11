from __future__ import annotations

COUNTRY_PROFILES: dict[str, dict[str, list[str]]] = {
    
    "Korea": {
        "macro_sensitivity": ["fed", "us_cpi", "usd", "treasury_yields", "oil", "global_risk_sentiment"],
        "key_sectors": ["semiconductors", "autos", "batteries", "shipbuilding", "defense", "airlines", "energy_imports"],
        "trade_links": ["us", "china", "japan", "taiwan", "middle_east", "global_supply_chain"],
        "geo_risks": ["north_korea", "china_taiwan", "middle_east", "global_trade", "shipping_routes"],
        "asset_focus": ["equities", "fx", "oil", "rates"]
    },

    "Japan": {
        "macro_sensitivity": ["boj", "usd_jpy", "oil", "treasury_yields", "global_risk_sentiment"],
        "key_sectors": ["autos", "electronics", "machinery", "energy_imports"],
        "trade_links": ["us", "china", "taiwan", "middle_east", "global_supply_chain"],
        "geo_risks": ["china_taiwan", "north_korea", "energy_supply", "shipping_routes"],
        "asset_focus": ["equities", "fx", "rates", "oil"]
    },

    "United States": {
        "macro_sensitivity": ["fed", "cpi", "labor_market", "treasury_yields", "oil", "global_risk_sentiment"],
        "key_sectors": ["technology", "financials", "energy", "defense", "consumer"],
        "trade_links": ["china", "europe", "japan", "middle_east", "global_supply_chain"],
        "geo_risks": ["middle_east", "china_taiwan", "russia", "global_trade", "sanctions"],
        "asset_focus": ["equities", "rates", "usd", "oil"]
    }
}