#!/usr/bin/env python3
"""Start a new edition, update or revision of the book.

Version numbers follow EDITION.UPDATE.REVISION:

    edition   2.0.0  A new edition: substantial rewrite or new chapters.
                     The edition number on the cover and title page goes up.
    update    1.1.0  Additions within the same edition (new sections, new figures).
    revision  1.0.1  Corrections only: typos, small fixes, a reprint.

Usage:
    python3 scripts/new_edition.py edition  --note "Adds a chapter on mixture of experts"
    python3 scripts/new_edition.py update   --note "New exercises in Part II"
    python3 scripts/new_edition.py revision --note "Fixes typos reported by readers"

It updates metadata.yaml (edition, version, date and the history list) and adds
a section to CHANGELOG.md. Then edit the manuscript, run `make all publish`,
commit, and tag the release: git tag v<version> && git push --tags
"""
import argparse
import datetime
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build import edition_label  # noqa: E402

META = os.path.join(ROOT, "metadata.yaml")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["edition", "update", "revision"])
    ap.add_argument("--note", default="", help="one-line summary for the edition history")
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    text = open(META, encoding="utf-8").read()
    meta = yaml.safe_load(text)
    major, minor, patch = (int(x) for x in str(meta["version"]).split("."))
    edition = int(meta["edition"])

    if args.kind == "edition":
        edition, major, minor, patch = edition + 1, major + 1, 0, 0
    elif args.kind == "update":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    version = f"{major}.{minor}.{patch}"
    if major != edition:
        sys.exit(f"Version {version} does not match edition {edition}; fix metadata.yaml first.")

    note = args.note or {"edition": f"{edition_label(edition)}.",
                         "update": "Additions and improvements.",
                         "revision": "Corrections."}[args.kind]

    text = re.sub(r"(?m)^edition:.*$", f"edition: {edition}", text, count=1)
    text = re.sub(r"(?m)^version:.*$", f"version: {version}", text, count=1)
    text = re.sub(r"(?m)^date:.*$", f"date: {args.date}", text, count=1)
    entry = (f"  - version: {version}\n    edition: {edition}\n"
             f"    date: {args.date}\n    note: {yaml.safe_dump(note, default_style=None).strip().splitlines()[0]}\n")
    text = text.rstrip("\n") + "\n" + entry
    yaml.safe_load(text)  # sanity check
    with open(META, "w", encoding="utf-8") as f:
        f.write(text)

    log = open(CHANGELOG, encoding="utf-8").read()
    head = f"## [{version}] — {edition_label(edition)} — {args.date}\n\n- {note}\n\n"
    marker = "<!-- new entries go below this line -->\n"
    log = log.replace(marker, marker + "\n" + head, 1) if marker in log else log + "\n" + head
    with open(CHANGELOG, "w", encoding="utf-8") as f:
        f.write(log)

    print(f"Now at {edition_label(edition)}, version {version} ({args.date}).")
    print("Next: edit manuscript/, then `make all publish`, commit, "
          f"`git tag v{version}` and `git push --tags`.")


if __name__ == "__main__":
    main()
