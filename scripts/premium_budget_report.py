#!/usr/bin/env python3
"""Print the shadow premium-request budget from the live route trace as JSON."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import premium_budget  # noqa: E402
from router.durable_decision_log import routes_path  # noqa: E402


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rows = []
    for line in routes_path().read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    print(json.dumps(premium_budget.report(rows, days=days), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
