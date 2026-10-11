"""
Stage 3c: Entity Resolution — CUDA, BATCHED + VECTORIZED clustering.

Fixes the bottleneck found in entity_resolution_cuda_batched.py: that version
computed similarity on GPU (fast) but then looped over every pair in plain
Python to find matches above the threshold (slow — O(n^2) Python iterations).

This version uses torch.where() to extract all above-threshold pairs directly
on the GPU/via vectorized tensor ops, so ONLY the matched pairs (a much
smaller list) are looped over in Python for the union-find step.

Also reports separate timings for:
  - embedding + similarity computation (the actual CUDA work)
  - clustering (union-find bookkeeping)
so the report can show the CUDA speedup cleanly, isolated from the
inherently-sequential bookkeeping step.
"""
from sentence_transformers import SentenceTransformer
import torch
import json
import time
import os

ENTITIES_PATH = "./data/entities.json"
OUTPUT_PATH = "./data/resolved_entities_cuda_vectorized.json"
SIMILARITY_THRESHOLD = 0.85


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on device: {device}")
    return SentenceTransformer("all-MiniLM-L6-v2", device=device), device


def main():
    model, device = load_model()

    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    all_texts = []
    for company, entities in all_entities.items():
        all_texts.extend([e["text"] for e in entities])

    unique_texts = list(set(all_texts))
    n = len(unique_texts)
    print(f"Running vectorized CUDA entity resolution: {n} unique entities")

    # --- Phase 1: embedding + similarity (the CUDA-heavy part) ---
    t0 = time.time()
    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
        batch_size=256,
    )
    norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    sim_matrix = norm @ norm.T  # (n, n) similarity matrix, on GPU

    # Zero out the lower triangle + diagonal so each pair is only counted once
    # and self-similarity doesn't count as a "match"
    upper_mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
    above_threshold = (sim_matrix >= SIMILARITY_THRESHOLD) & upper_mask

    # Extract matching pair indices directly via GPU-vectorized op
    match_indices = torch.nonzero(above_threshold, as_tuple=False)  # (num_matches, 2)
    match_indices_cpu = match_indices.cpu().numpy()

    embed_sim_time = time.time() - t0
    print(f"Embedding + similarity (CUDA) time: {embed_sim_time:.3f}s")
    print(f"Matching pairs found: {len(match_indices_cpu)}")

    # --- Phase 2: clustering, now only over actual matches (small list) ---
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

    for i, j in match_indices_cpu:
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
    print(f"\nVectorized CUDA entity resolution done.")
    print(f"  Embedding + similarity (CUDA): {embed_sim_time:.3f}s")
    print(f"  Clustering (CPU bookkeeping):  {clustering_time:.3f}s")
    print(f"  TOTAL:                          {total_time:.3f}s")
    print(f"{len(unique_texts)} unique mentions -> {num_unique_resolved} resolved entities")
    print(f"Results saved to {OUTPUT_PATH}")
    return total_time


if __name__ == "__main__":
    main()
