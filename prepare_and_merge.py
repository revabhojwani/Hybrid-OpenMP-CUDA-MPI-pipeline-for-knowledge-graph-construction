"""
Two helper functions for the OpenMP NER pipeline:

1. prepare_file_list() - generates the file_list.txt the C++ dispatcher reads
2. merge_results()     - combines all batch_N_result.json files into one
                          entities.json, matching the sequential baseline's
                          output format (for direct comparison / accuracy check)

Usage:
    python prepare_and_merge.py prepare
    python prepare_and_merge.py merge
"""
import json
import sys
import os
from pathlib import Path

TEXT_DIR = "./data/raw_text"
FILE_LIST_PATH = "./openmp_output/file_list.txt"
OUTPUT_DIR = "./openmp_output"
MERGED_OUTPUT = "./data/entities_openmp.json"


def prepare_file_list():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text_files = sorted(Path(TEXT_DIR).glob("*.txt"))
    abs_paths = [str(p.resolve()) for p in text_files]
    with open(FILE_LIST_PATH, "w") as f:
        f.write("\n".join(abs_paths))
    print(f"Wrote {len(abs_paths)} file paths to {FILE_LIST_PATH}")


def merge_results():
    merged = {}
    batch_files = sorted(Path(OUTPUT_DIR).glob("batch_*_result.json"))
    if not batch_files:
        print("No batch result files found — did the OpenMP run complete?")
        return

    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        merged.update(batch_data)

    os.makedirs(os.path.dirname(MERGED_OUTPUT), exist_ok=True)
    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(batch_files)} batch files -> {len(merged)} companies")
    print(f"Saved to {MERGED_OUTPUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "prepare":
        prepare_file_list()
    elif mode == "merge":
        merge_results()
    else:
        print("Usage: python prepare_and_merge.py [prepare|merge]")
