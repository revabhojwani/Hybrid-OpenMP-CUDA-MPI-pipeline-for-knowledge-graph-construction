"""
Stage 2: Relation Extraction — SEQUENTIAL baseline.
Uses a dependency-parsing approach: for each sentence, find subject-verb-object
patterns connecting two named entities. This is a lightweight, explainable
method (no LLM calls needed), good enough for a parallel-computing-focused
project where RE is not the main research contribution.
"""
import spacy
import json
import time
import os
from pathlib import Path

TEXT_DIR = "./data/raw_text"
ENTITIES_PATH = "./data/entities.json"
OUTPUT_PATH = "./data/relations.json"


def load_model():
    return spacy.load("en_core_web_sm")


def extract_relations(nlp, text, entities, max_chars=200_000):
    """
    For each sentence, look for a verb connecting two entity mentions.
    Simple SVO-style extraction using spaCy's dependency parse.
    """
    text = text[:max_chars]
    doc = nlp(text)

    # Build a quick lookup: char position -> entity text
    entity_spans = [(e["start"], e["end"], e["text"]) for e in entities]

    relations = []
    for sent in doc.sents:
        sent_start, sent_end = sent.start_char, sent.end_char
        # entities that fall within this sentence
        sent_entities = [
            (s, e, t) for (s, e, t) in entity_spans
            if s >= sent_start and e <= sent_end
        ]
        if len(sent_entities) < 2:
            continue

        # find the root verb of the sentence
        root = sent.root
        if root.pos_ != "VERB":
            continue

        # naive pairing: first two entities in sentence, connected by root verb
        for i in range(len(sent_entities) - 1):
            e1 = sent_entities[i]
            e2 = sent_entities[i + 1]
            relations.append({
                "subject": e1[2],
                "relation": root.lemma_,
                "object": e2[2],
                "sentence": sent.text.strip()[:200],
            })

    return relations


def main():
    nlp = load_model()

    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    text_files = sorted(Path(TEXT_DIR).glob("*.txt"))
    print(f"Running sequential relation extraction on {len(text_files)} documents...")

    results = {}
    start = time.time()

    for i, path in enumerate(text_files, 1):
        company = path.stem
        if company not in all_entities:
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        t0 = time.time()
        relations = extract_relations(nlp, text, all_entities[company])
        t1 = time.time()

        results[company] = relations
        print(f"[{i}/{len(text_files)}] {company}: {len(relations)} relations ({t1 - t0:.2f}s)")

    elapsed = time.time() - start
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSequential relation extraction done. Total time: {elapsed:.2f}s")
    print(f"Results saved to {OUTPUT_PATH}")
    return elapsed


if __name__ == "__main__":
    main()
