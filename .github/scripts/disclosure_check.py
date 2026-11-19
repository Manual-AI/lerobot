#!/usr/bin/env python3
"""Scan text and git history for terms that must not appear in this public repository.

The list of terms is NOT stored in the repository (it would defeat the purpose). It is read from:
  1. the file given by --denylist-file, or
  2. the DISCLOSURE_DENYLIST environment variable (CI supplies it from a repository secret).
One case-insensitive regular expression per line; blank lines and lines starting with '#' are ignored.

Usage:
  disclosure_check.py --stdin [--label NAME]          scan text from stdin
  disclosure_check.py --git-range BASE..HEAD          scan commit messages and added lines
Matched text is never printed (CI logs are public); only the location and pattern number are.
Exit status: 0 clean, 1 match found, 2 no denylist available.
"""

import argparse
import os
import re
import subprocess
import sys


def load_patterns(path):
    raw = open(path).read() if path else os.environ.get("DISCLOSURE_DENYLIST", "")
    return [
        (i, re.compile(line, re.IGNORECASE))
        for i, line in enumerate((ln.strip() for ln in raw.splitlines()), 1)
        if line and not line.startswith("#")
    ]


def scan(text, label, patterns):
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for idx, pat in patterns:
            if pat.search(line):
                hits.append(f"{label}:{lineno}: matches denylist pattern #{idx}")
    return hits


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--denylist-file")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--label", default="text")
    ap.add_argument("--git-range")
    args = ap.parse_args()

    patterns = load_patterns(args.denylist_file)
    if not patterns:
        print("No denylist available (set DISCLOSURE_DENYLIST or --denylist-file).", file=sys.stderr)
        return 2

    hits = []
    if args.stdin:
        hits += scan(sys.stdin.read(), args.label, patterns)
    if args.git_range:
        base, _, head = args.git_range.partition("..")
        for sha in git("rev-list", args.git_range).split():
            hits += scan(git("log", "-1", "--format=%B", sha), f"commit {sha[:9]} message", patterns)
        current = None
        for line in git("diff", "-U0", "--no-color", f"{base}...{head}").splitlines():
            if line.startswith("+++ b/"):
                current = line[6:]
            elif line.startswith("+") and not line.startswith("+++"):
                hits += scan(line[1:], f"added line in {current}", patterns)

    for h in hits:
        print(h)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
