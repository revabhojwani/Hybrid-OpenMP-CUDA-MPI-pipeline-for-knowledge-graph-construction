"""
Creates smaller file lists (10 and 40 companies) for scalability testing,
so you can rerun the OpenMP NER pipeline at multiple corpus sizes and
show how speedup changes with scale -- directly supports the paper's
"at what size does parallelization pay off" research question.

Usage:
    python make_scale_lists.py
"""
from pathlib import Path
import os

TEXT_DIR = "./data/raw_text"
OUTPUT_DIR = "./openmp_output"


def main():
    text_files = sorted(Path(TEXT_DIR).glob("*.txt"))
    abs_paths = [str(p.resolve()) for p in text_files]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for n in [10, 40]:
        subset = abs_paths[:n]
        out_path = f"{OUTPUT_DIR}/file_list_{n}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(subset))
        print(f"Wrote {len(subset)} file paths to {out_path}")

    print(f"\n(Full 79-company list already exists at {OUTPUT_DIR}/file_list.txt)")


if __name__ == "__main__":
    main()
