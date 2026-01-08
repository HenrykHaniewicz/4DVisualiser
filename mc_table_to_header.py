"""
Convert a Python Marching Cubes TRI_TABLE (list of lists, -1-terminated rows)
into a fixed 256x16 C++ int table padded with -1.

Usage:
  python generate_mc_tri_table.py > marching_cubes_tri_table.h
or:
  python generate_mc_tri_table.py --out marching_cubes_tri_table.h
"""

import argparse
import re
import sys
from typing import List

def parse_python_tri_table(text: str) -> List[List[int]]:
    # Find the start of "TRI_TABLE = ["
    m = re.search(r"\bTRI_TABLE\b\s*=\s*\[", text)
    if not m:
        raise ValueError("Could not find 'TRI_TABLE = [' in input.")

    i = m.end()  # position right after the opening '['
    depth = 1
    start = i
    n = len(text)

    # Scan forward to find the matching closing ']' for TRI_TABLE
    while i < n and depth > 0:
        c = text[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
        i += 1

    if depth != 0:
        raise ValueError("Unbalanced brackets while scanning TRI_TABLE.")

    body = text[start:i-1]  # content inside TRI_TABLE's outer brackets

    # Extract rows: [ ... ] (these are not nested in the MC table)
    row_strs = re.findall(r"\[([^\[\]]*)\]", body, flags=re.DOTALL)
    if not row_strs:
        raise ValueError("Found TRI_TABLE, but no row lists like '[...]' were parsed.")

    rows: List[List[int]] = []
    for rs in row_strs:
        parts = [p.strip() for p in rs.split(",") if p.strip()]
        rows.append([int(p) for p in parts])

    return rows


def normalize_row_to_16(row: List[int]) -> List[int]:
    """
    Ensures row contains -1 terminator and pads/truncates to exactly 16 entries.
    Marching Cubes triangle rows are up to 16 ints (15 indices + -1).
    """
    # If row already contains -1, cut at first -1 (inclusive)
    if -1 in row:
        row = row[: row.index(-1) + 1]
    else:
        # If user omitted -1 terminator, add it
        row = row + [-1]

    if len(row) > 16:
        raise ValueError(f"Row longer than 16 after termination: {row} (len={len(row)})")

    # Pad with -1 to 16
    row = row + [-1] * (16 - len(row))
    return row


def emit_cpp_header(rows_256x16: List[List[int]]) -> str:
    lines: List[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("// Generated Marching Cubes triangle table")
    lines.append("// 256 cases x 16 entries, padded with -1")
    lines.append("static const int TRI_TABLE[256][16] = {")
    for r in rows_256x16:
        lines.append("    {" + ",".join(str(v) for v in r) + "},")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="infile",
        default="-",
        help="Input file containing TRI_TABLE = [...] (default: stdin)",
    )
    ap.add_argument(
        "--out",
        dest="outfile",
        default="-",
        help="Output header file path (default: stdout)",
    )
    args = ap.parse_args()

    # Read input
    if args.infile == "-" or args.infile == "":
        text = sys.stdin.read()
    else:
        with open(args.infile, "r", encoding="utf-8") as f:
            text = f.read()

    rows = parse_python_tri_table(text)

    if len(rows) != 256:
        raise ValueError(f"Expected 256 rows, got {len(rows)}.")

    fixed: List[List[int]] = [normalize_row_to_16(r) for r in rows]

    header = emit_cpp_header(fixed)

    # Write output
    if args.outfile == "-" or args.outfile == "":
        sys.stdout.write(header)
    else:
        with open(args.outfile, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
