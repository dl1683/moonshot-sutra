"""Opsec regression check — scans tracked files for banned model names.

Prevents accidental re-introduction of specific teacher model identities
into public-facing code or documentation.

Usage:
    python check_opsec.py          # exit 0 = clean, exit 1 = violations found
    python check_opsec.py --fix    # show suggested replacements
"""

import re
import subprocess
import sys

BANNED_PATTERNS = [
    r"[Qq]wen3?[-_]\d+(\.\d+)?[Bb]",
    r"[Qq]wen/[Qq]wen",
    r"\bLFM[\d\s(]",
    r"[Ll][Ff][Mm]2\.?5[-_]1[._]2[Bb]",
    r"[Ll]iquid\s*AI/",
    r"[Ee]mbedding[Gg]emma",
    r"[Gg]emma[-_]\d",
    r"google/embedding",
    r"[Mm]amba\d?[-_]\d",
    r"state-spaces/mamba",
    r"qwen3_[01]p[67]b",
    r"lfm2_1p2b",
    r"mamba2_780m",
    r"embeddinggemma_300m",
]

ALLOWED_FILES = {
    ".gitignore",
    "config/teacher_config.json",
    "code/check_opsec.py",
    "code/test_utilities.py",
}

COMPILED = [re.compile(p) for p in BANNED_PATTERNS]


def get_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: git ls-files failed")
        sys.exit(2)
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def scan_file(path: str) -> list[tuple[int, str, str]]:
    violations = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for pattern in COMPILED:
                    match = pattern.search(line)
                    if match:
                        violations.append((lineno, match.group(), line.rstrip()))
                        break
    except (OSError, UnicodeDecodeError):
        pass
    return violations


def scan_git_history() -> list[tuple[str, str, str]]:
    """Scan all commit subjects and bodies for banned patterns."""
    result = subprocess.run(
        ["git", "log", "--format=%H\t%s\t%b", "--all"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: git log failed, skipping history scan")
        return []
    violations = []
    for line in result.stdout.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        commit_hash = parts[0][:12]
        message = "\t".join(parts[1:])
        for pattern in COMPILED:
            match = pattern.search(message)
            if match:
                violations.append((commit_hash, match.group(), message[:120]))
                break
    return violations


def main():
    show_fix = "--fix" in sys.argv
    check_history = "--history" in sys.argv

    files = get_tracked_files()
    total_violations = 0
    violation_files = []

    for path in files:
        normalized = path.replace("\\", "/")
        if normalized in ALLOWED_FILES:
            continue

        hits = scan_file(path)
        if hits:
            violation_files.append(path)
            total_violations += len(hits)
            print(f"\n  {path}:")
            for lineno, matched, line in hits:
                print(f"    L{lineno}: matched '{matched}'")
                if show_fix:
                    print(f"      > {line[:120]}")

    history_violations = []
    if check_history:
        history_violations = scan_git_history()
        if history_violations:
            print(f"\nHISTORY VIOLATIONS ({len(history_violations)} commits):")
            for commit, matched, msg in history_violations[:20]:
                print(f"  {commit}: matched '{matched}' in: {msg[:80]}")
            if len(history_violations) > 20:
                print(f"  ... and {len(history_violations) - 20} more")

    if total_violations:
        print(f"\nOPSEC FAIL: {total_violations} banned model name(s) "
              f"in {len(violation_files)} tracked file(s).")
        print("Use role aliases (t0_anchor_decoder, etc.) instead.")
        return 1
    elif history_violations:
        print(f"\nOPSEC WARNING: tracked files clean, but {len(history_violations)} "
              f"commit message(s) contain banned names.")
        print("Squash or rewrite history before pushing to public remote.")
        return 2
    else:
        msg = "OPSEC OK: no banned model names in tracked files."
        if check_history:
            msg = "OPSEC OK: no banned model names in tracked files or git history."
        print(msg)
        return 0


if __name__ == "__main__":
    sys.exit(main())
