"""
Merges batch_N_result.json files from the OpenMP relation extraction run
into one relations_openmp.json, matching the sequential baseline's format
for direct comparison.

Usage:
    python merge_re_results.py <output_dir>
"""
import json
import sys
from pathlib import Path

MERGED_OUTPUT = "./data/relations_openmp.json"


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./re_openmp_output"

    merged = {}
    batch_files = sorted(Path(output_dir).glob("batch_*_result.json"))
    if not batch_files:
        print(f"No batch result files found in {output_dir}")
        return

    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        merged.update(batch_data)

    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(batch_files)} batch files -> {len(merged)} companies")
    print(f"Saved to {MERGED_OUTPUT}")


if __name__ == "__main__":
    main()
