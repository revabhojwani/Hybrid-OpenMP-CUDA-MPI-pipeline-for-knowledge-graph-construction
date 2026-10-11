"""
Creates subset versions of entities.json, relations.json for the first
10 and 40 companies (alphabetically, matching make_scale_lists.py's
selection), so entity resolution (CUDA) and graph population (MPI)
can be scalability-tested the same way NER was.

Usage:
    python make_scale_data.py
"""
import json
import os

ENTITIES_PATH = "./data/entities.json"
RELATIONS_PATH = "./data/relations.json"
OUTPUT_DIR = "./data"


def main():
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)
    with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
        all_relations = json.load(f)

    companies_sorted = sorted(all_entities.keys())

    for n in [10, 40]:
        subset_companies = companies_sorted[:n]

        subset_entities = {c: all_entities[c] for c in subset_companies if c in all_entities}
        subset_relations = {c: all_relations[c] for c in subset_companies if c in all_relations}

        with open(f"{OUTPUT_DIR}/entities_{n}.json", "w", encoding="utf-8") as f:
            json.dump(subset_entities, f, indent=2)
        with open(f"{OUTPUT_DIR}/relations_{n}.json", "w", encoding="utf-8") as f:
            json.dump(subset_relations, f, indent=2)

        print(f"n={n}: wrote entities_{n}.json ({len(subset_entities)} companies), "
              f"relations_{n}.json ({len(subset_relations)} companies)")


if __name__ == "__main__":
    main()
