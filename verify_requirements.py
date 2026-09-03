"""
Checks every 'package==version' line in a requirements file against the
live PyPI JSON API and reports any package/version that doesn't exist.

Usage:
    pip install requests
    python verify_requirements.py requirements_clean.txt
"""

import sys
import re
import requests

def check_file(path):
    bad = []
    ok = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for line in lines:
        m = re.match(r"^([A-Za-z0-9._-]+)==([A-Za-z0-9.+!_-]+)$", line)
        if not m:
            print(f"SKIP (couldn't parse): {line}")
            continue
        name, version = m.group(1), m.group(2)
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ok += 1
            else:
                bad.append((name, version, r.status_code))
                print(f"MISSING: {name}=={version}  (HTTP {r.status_code})")
        except requests.RequestException as e:
            bad.append((name, version, str(e)))
            print(f"ERROR checking {name}=={version}: {e}")

    print(f"\nChecked {len(lines)} lines. OK: {ok}  Problems: {len(bad)}")
    if bad:
        print("\nPackages needing attention:")
        for name, version, reason in bad:
            print(f"  - {name}=={version}  ({reason})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_requirements.py <requirements_file>")
        sys.exit(1)
    check_file(sys.argv[1])
