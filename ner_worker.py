"""
NER worker — processes ONE batch of documents.
Called by the OpenMP C++ dispatcher (one call per thread, per batch),
not by the user directly.

Usage:
    python ner_worker.py <file_list.txt> <output.json>

<file_list.txt> contains one .txt file path per line (this thread's batch).
"""
import spacy
import json
import sys
import time

RELEVANT_LABELS = {"ORG", "PERSON", "GPE", "MONEY", "DATE", "PRODUCT", "LAW"}


def extract_entities(nlp, text, max_chars=200_000):
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
    file_list_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(file_list_path, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]

    nlp = spacy.load("en_core_web_sm")
    results = {}

    for path in file_paths:
        company = path.split("/")[-1].replace(".txt", "")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        results[company] = extract_entities(nlp, text)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
