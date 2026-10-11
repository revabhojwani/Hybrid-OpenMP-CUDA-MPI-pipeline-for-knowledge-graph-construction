"""
Stage 1: Named Entity Recognition — SEQUENTIAL baseline.
Processes each company's extracted text one at a time, one document at a time.
This is the version OpenMP will later parallelize (across documents).
"""
import spacy
import json
import time
import os
from pathlib import Path

TEXT_DIR = "./data/raw_text"
OUTPUT_PATH = "./data/entities.json"

# Entity types relevant to financial/company text
RELEVANT_LABELS = {"ORG", "PERSON", "GPE", "MONEY", "DATE", "PRODUCT", "LAW"}


def load_model():
    return spacy.load("en_core_web_sm")


def extract_entities(nlp, text, max_chars=200_000):
    """Run NER on text, return list of entities with type + position."""
    # spaCy has a max doc length; truncate very long filings
    text = text[:max_chars]
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return entities


def main():
    nlp = load_model()
    text_files = sorted(Path(TEXT_DIR).glob("*.txt"))
    print(f"Running sequential NER on {len(text_files)} documents...")

    results = {}
    start = time.time()

    for i, path in enumerate(text_files, 1):
        company = path.stem
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        t0 = time.time()
        entities = extract_entities(nlp, text)
        t1 = time.time()

        results[company] = entities
        print(f"[{i}/{len(text_files)}] {company}: {len(entities)} entities ({t1 - t0:.2f}s)")

    elapsed = time.time() - start
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSequential NER done. Total time: {elapsed:.2f}s")
    print(f"Results saved to {OUTPUT_PATH}")
    return elapsed


if __name__ == "__main__":
    main()
