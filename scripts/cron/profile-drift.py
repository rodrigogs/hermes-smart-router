#!/usr/bin/env python3
"""Report profile <-> router.yaml <-> provider-cache drift; silent when clean.

Intended for a no_agent cron with ``monitor``: output is deterministic (sorted,
no timestamps), so an unchanged drift never re-notifies. Exit 0 always.

Proposed (NOT created) cron; the script must live under ~/.hermes/scripts/:
  ln -s $PWD/scripts/cron/profile-drift.py ~/.hermes/scripts/profile-drift.py
  hermes cron create "every 6h" --name profile-drift --no-agent --deliver local \
    --script profile-drift.py --monitor-script profile-drift.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from router import paths, profile_drift  # noqa: E402


def main() -> int:
    root = paths.hermes_root()
    plugin = Path(__file__).resolve().parents[2]
    lines = profile_drift.drift(
        paths.resolve_policy_path(plugin), root / "profiles",
        root / "provider_models_cache.json", root,
    )
    if lines:
        print(f"profile drift: {len(lines)} divergence(s)")
        for line in lines:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
