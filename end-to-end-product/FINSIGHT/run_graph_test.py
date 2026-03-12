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

    print("\n=== LOGS ===\n")
    pprint(result.get("logs", []))


if __name__ == "__main__":
    main()