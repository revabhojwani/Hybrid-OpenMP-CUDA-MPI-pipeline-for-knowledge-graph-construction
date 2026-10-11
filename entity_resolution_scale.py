"""
Stage 3 — Entity Resolution scalability test.
Same batched + vectorized logic as entity_resolution_cuda_vectorized.py /
entity_resolution_sequential_batched.py, but parameterized to run on any
entities JSON file (e.g. entities_10.json, entities_40.json, entities.json)
and either CPU or CUDA, so you can build a corpus-size scaling table.

Usage:
    python entity_resolution_scale.py <entities.json> <cpu|cuda>
"""
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import time
import sys

SIMILARITY_THRESHOLD = 0.85


def main():
    entities_path = sys.argv[1]
    mode = sys.argv[2]  # "cpu" or "cuda"

    device = "cuda" if mode == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Running entity resolution on: {device} ({entities_path})")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    with open(entities_path, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    all_texts = []
    for company, entities in all_entities.items():
        all_texts.extend([e["text"] for e in entities])

    unique_texts = list(set(all_texts))
    n = len(unique_texts)
    print(f"Companies: {len(all_entities)}, unique entities: {n}")

    t0 = time.time()
    if device == "cuda":
        embeddings = model.encode(unique_texts, convert_to_tensor=True,
                                   device=device, show_progress_bar=False, batch_size=256)
        norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        sim_matrix = norm @ norm.T
        upper_mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
        above_threshold = (sim_matrix >= SIMILARITY_THRESHOLD) & upper_mask
        match_indices = torch.nonzero(above_threshold, as_tuple=False).cpu().numpy()
    else:
        embeddings = np.array(model.encode(unique_texts, show_progress_bar=False, batch_size=256))
        norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim_matrix = norm @ norm.T
        upper_mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        above_threshold = (sim_matrix >= SIMILARITY_THRESHOLD) & upper_mask
        match_indices = np.argwhere(above_threshold)

    embed_sim_time = time.time() - t0

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

    resolved = set(find(i) for i in range(n))
    clustering_time = time.time() - t1

    total = embed_sim_time + clustering_time
    print(f"Embedding+similarity: {embed_sim_time:.3f}s | Clustering: {clustering_time:.3f}s | Total: {total:.3f}s")
    print(f"{n} unique mentions -> {len(resolved)} resolved entities")


if __name__ == "__main__":
    main()
