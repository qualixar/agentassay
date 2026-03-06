#!/usr/bin/env python3
"""Add Qualixar attribution headers to all AgentAssay source files.

This script adds the standard 4-line header to every .py file in src/agentassay/
that doesn't already have it. The header is inserted BEFORE any existing
docstrings, comments, or imports.
"""

from pathlib import Path

HEADER = """# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""


def has_header(content: str) -> bool:
    """Check if file already has the Qualixar header."""
    return "Part of Qualixar" in content[:500]


def add_header_to_file(file_path: Path) -> bool:
    """Add header to a single file if it doesn't have one.

    Returns
    -------
    bool
        True if header was added, False if already present.
    """
    content = file_path.read_text(encoding="utf-8")

    if has_header(content):
        return False

    # Add header at the beginning
    new_content = HEADER + content
    file_path.write_text(new_content, encoding="utf-8")
    return True


def main() -> None:
    """Add headers to all Python files in src/agentassay/."""
    src_dir = Path(__file__).parent.parent / "src" / "agentassay"

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        return

    # Find all .py files
    py_files = list(src_dir.rglob("*.py"))

    added_count = 0
    skipped_count = 0

    for py_file in py_files:
        if add_header_to_file(py_file):
            print(f"✓ Added header to {py_file.relative_to(src_dir.parent.parent)}")
            added_count += 1
        else:
            skipped_count += 1

    print(f"\nSummary:")
    print(f"  Headers added: {added_count}")
    print(f"  Already present: {skipped_count}")
    print(f"  Total files: {len(py_files)}")


if __name__ == "__main__":
    main()
