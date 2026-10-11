"""
Stage 0: Extract text from 10-K PDF filings.
Run once — caches extracted text as .txt files so later stages don't
need to re-parse PDFs every time.
"""
import pdfplumber
import os
import time
from pathlib import Path

PDF_DIR = "/mnt/c/Users/reva/Internship/10k"
OUTPUT_DIR = "./data/raw_text"

# Limit pages per doc for speed during development.
# 10-Ks are huge (100-300+ pages) but the NLP-relevant sections
# (Business, Risk Factors, MD&A) are usually in the first ~60 pages.
MAX_PAGES = 60


def extract_pdf(pdf_path, max_pages=MAX_PAGES):
    """Extract text from a single PDF, page by page."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = min(len(pdf.pages), max_pages)
        for i in range(num_pages):
            page_text = pdf.pages[i].extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    start = time.time()
    for i, pdf_path in enumerate(pdf_files, 1):
        company_name = pdf_path.stem
        out_path = os.path.join(OUTPUT_DIR, f"{company_name}.txt")

        if os.path.exists(out_path):
            print(f"[{i}/{len(pdf_files)}] {company_name} — already extracted, skipping")
            continue

        print(f"[{i}/{len(pdf_files)}] Extracting {company_name}...")
        try:
            text = extract_pdf(pdf_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"    -> {len(text)} characters extracted")
        except Exception as e:
            print(f"    !! FAILED: {e}")

    elapsed = time.time() - start
    print(f"\nDone. Total extraction time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
