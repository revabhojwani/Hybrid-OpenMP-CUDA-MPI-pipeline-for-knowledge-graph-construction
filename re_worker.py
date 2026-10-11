"""
Relation Extraction worker — processes ONE batch of documents.
Called by the OpenMP C++ dispatcher, same pattern as ner_worker.py.

Usage:
    python re_worker.py <file_list.txt> <entities.json> <output.json>
"""
import spacy
import json
import sys


def extract_relations(nlp, text, entities, max_chars=200_000):
    text = text[:max_chars]
    doc = nlp(text)
    entity_spans = [(e["start"], e["end"], e["text"]) for e in entities]

    relations = []
    for sent in doc.sents:
        sent_start, sent_end = sent.start_char, sent.end_char
        sent_entities = [
            (s, e, t) for (s, e, t) in entity_spans
            if s >= sent_start and e <= sent_end
        ]
        if len(sent_entities) < 2:
            continue

        root = sent.root
        if root.pos_ != "VERB":
            continue

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
    file_list_path = sys.argv[1]
    entities_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(file_list_path, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]

    with open(entities_path, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    nlp = spacy.load("en_core_web_sm")
    results = {}

    for path in file_paths:
        company = path.split("/")[-1].replace(".txt", "")
        if company not in all_entities:
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        results[company] = extract_relations(nlp, text, all_entities[company])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
