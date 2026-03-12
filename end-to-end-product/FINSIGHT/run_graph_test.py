from datetime import date
from pprint import pprint

from app.graph.workflow import build_workflow


def main() -> None:
    workflow = build_workflow()

    initial_state = {
        "date": date(2026, 3, 10),
        "country": "Korea",
        "max_items_per_worker": 5,
        "report_type": "eod",
        "logs": [],
        "errors": [],
    }

    result = workflow.invoke(initial_state)

    print("\n=== REPORT ===\n")
    print(result["report_markdown"])

    print("\n=== TOP EVENT ===\n")
    if result.get("ranked_events"):
        pprint(result["ranked_events"][0].model_dump())

    print("\n=== EVENT COUNTS ===\n")
    print("macro:", len(result.get("macro_events", [])))
    print("markets:", len(result.get("markets_events", [])))
    print("commodities_fx:", len(result.get("commodities_fx_events", [])))
    print("geopolitical:", len(result.get("geopolitical_events", [])))


    print("\n=== MACRO EVENTS ===\n")
    for event in result.get("macro_events", []):
        print(f"{event.source} | {event.headline}")

    print("\n=== MARKETS EVENTS ===\n")
    for event in result.get("markets_events", []):
        print(f"{event.source} | {event.headline}")

    print("\n=== COMMODITIES & FX EVENTS ===\n")
    for event in result.get("commodities_fx_events", []):
        print(f"{event.source} | {event.headline}")

    print("\n=== GEOPOLITICAL EVENTS ===\n")
    for event in result.get("geopolitical_events", []):
        print(f"{event.source} | {event.headline}")

    print("\n=== LOGS ===\n")
    pprint(result.get("logs", []))


if __name__ == "__main__":
    main()