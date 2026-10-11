"""
Stage 3 — Sequential baseline, BATCHED (fair comparison version).

Same global resolution logic as entity_resolution_cuda_vectorized.py
(all companies' entities combined into one resolution pass, vectorized
pair extraction via NumPy instead of a raw Python double-loop), but
running on CPU only, no GPU involved.

This exists purely so entity_resolution_cuda_vectorized.py has a fair,
apples-to-apples baseline to compare against -- same task, same
algorithm, only the embedding + similarity computation differs
(CPU/NumPy here vs GPU/CUDA there).
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time
import os

ENTITIES_PATH = "./data/entities.json"
OUTPUT_PATH = "./data/resolved_entities_sequential_batched.json"
SIMILARITY_THRESHOLD = 0.85


def load_model():
    print("Loading embedding model on device: cpu")
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def main():
    model = load_model()

    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    all_texts = []
    for company, entities in all_entities.items():
        all_texts.extend([e["text"] for e in entities])

    unique_texts = list(set(all_texts))
    n = len(unique_texts)
    print(f"Running sequential (CPU) batched entity resolution: {n} unique entities")

    # --- Phase 1: embedding + similarity (CPU/NumPy) ---
    t0 = time.time()
    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        batch_size=256,
    )
    embeddings = np.array(embeddings)
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = norm @ norm.T  # (n, n), CPU/NumPy matrix multiply

    # Same vectorized pair-extraction approach as the CUDA version,
    # just using NumPy instead of torch, to keep the algorithm identical
    upper_mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    above_threshold = (sim_matrix >= SIMILARITY_THRESHOLD) & upper_mask
    match_indices = np.argwhere(above_threshold)

    embed_sim_time = time.time() - t0
    print(f"Embedding + similarity (CPU) time: {embed_sim_time:.3f}s")
    print(f"Matching pairs found: {len(match_indices)}")

    # --- Phase 2: clustering (identical logic to CUDA version) ---
    t1 = time.time()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in match_indices:
        union(int(i), int(j))

    clusters = {}
    for i, text in enumerate(unique_texts):
        root = find(i)
        canonical = unique_texts[root]
        clusters.setdefault(canonical, []).append(text)

    mapping = {}
    for canonical, variants in clusters.items():
        for v in variants:
            mapping[v] = canonical

    clustering_time = time.time() - t1
    print(f"Clustering (union-find) time: {clustering_time:.3f}s")

    total_time = embed_sim_time + clustering_time

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    num_unique_resolved = len(set(mapping.values()))
    print(f"\nSequential batched entity resolution done.")
    print(f"  Embedding + similarity (CPU): {embed_sim_time:.3f}s")
    print(f"  Clustering (CPU bookkeeping): {clustering_time:.3f}s")
    print(f"  TOTAL:                         {total_time:.3f}s")
    print(f"{len(unique_texts)} unique mentions -> {num_unique_resolved} resolved entities")
    print(f"Results saved to {OUTPUT_PATH}")
    return total_time


if __name__ == "__main__":
    main()
