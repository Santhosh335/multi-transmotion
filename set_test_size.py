"""
Sets the test-split truncation size in extract_data/NBA/extract_NBA.py,
without touching the train or val branches (which may currently read
the same number as test, so a plain find-replace isn't safe here).

Usage:  python set_test_size.py 1500
"""
import re
import sys

path = "extract_data/NBA/extract_NBA.py"
n = sys.argv[1]

with open(path, "r") as f:
    lines = f.readlines()

pattern = re.compile(r"array_data\[:\d+\]")

for i, line in enumerate(lines):
    if "elif args.split == 'test':" in line:
        # The truncation is on the next line
        next_line = lines[i + 1]
        if not pattern.search(next_line):
            print("ERROR: pattern not found on the expected line - no change made.")
            sys.exit(1)
        lines[i + 1] = pattern.sub(f"array_data[:{n}]", next_line)
        break
else:
    print("ERROR: could not find the test branch in the file.")
    sys.exit(1)

with open(path, "w") as f:
    f.writelines(lines)

print(f"Test truncation set to {n}: {lines[i+1].strip()}")
