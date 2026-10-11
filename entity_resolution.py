"""
Stage 3: Entity Resolution — SEQUENTIAL baseline.
Groups entity mentions that refer to the same real-world entity
(e.g. "Apple" and "Apple Inc." -> same node) using embedding similarity.

This is the stage that will later be accelerated with CUDA, since
it involves computing pairwise similarity across many entity embeddings —
repetitive batch math, ideal for GPU.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time
import os

ENTITIES_PATH = "./data/entities.json"
OUTPUT_PATH = "./data/resolved_entities.json"
SIMILARITY_THRESHOLD = 0.85


def load_model():
    # Small, fast embedding model — fine for CPU baseline.
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_sim_matrix(embeddings):
    """Compute pairwise cosine similarity, sequential (numpy) version."""
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return norm @ norm.T


def resolve_entities(model, entity_texts):
    """
    Cluster entity strings that are semantically the same, using
    embedding similarity + a simple threshold-based union-find.
    """
    unique_texts = list(set(entity_texts))
    if len(unique_texts) == 0:
        return {}

    embeddings = model.encode(unique_texts, show_progress_bar=False)
    sim_matrix = cosine_sim_matrix(np.array(embeddings))

    # union-find for clustering
    parent = list(range(len(unique_texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    n = len(unique_texts)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                union(i, j)

    # build clusters: canonical name -> list of variant names
    clusters = {}
    for i, text in enumerate(unique_texts):
        root = find(i)
        canonical = unique_texts[root]
        clusters.setdefault(canonical, []).append(text)

    # mapping: every variant -> canonical form
    mapping = {}
    for canonical, variants in clusters.items():
        for v in variants:
            mapping[v] = canonical

    return mapping


def main():
    model = load_model()

    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    print(f"Running sequential entity resolution on {len(all_entities)} documents...")

    results = {}
    start = time.time()

    for i, (company, entities) in enumerate(all_entities.items(), 1):
        entity_texts = [e["text"] for e in entities]

        t0 = time.time()
        mapping = resolve_entities(model, entity_texts)
        t1 = time.time()

        results[company] = mapping
        num_unique = len(set(mapping.values()))
        print(f"[{i}/{len(all_entities)}] {company}: "
              f"{len(entity_texts)} mentions -> {num_unique} resolved entities ({t1 - t0:.2f}s)")

    elapsed = time.time() - start
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSequential entity resolution done. Total time: {elapsed:.2f}s")
    print(f"Results saved to {OUTPUT_PATH}")
    return elapsed


if __name__ == "__main__":
    main()
