"""
Stage 3b: Entity Resolution — CUDA, BATCHED across all companies.

Same GPU-accelerated logic as entity_resolution_cuda.py, but instead of
running a separate small GPU call per company (79 small calls, each paying
GPU transfer/launch overhead), this combines ALL companies' entities into
ONE large batch and resolves them together in a single GPU pass.

This tests whether CUDA's advantage appears once the batch size is large
enough to outweigh per-call overhead -- directly answers the "does GPU
acceleration help at this workload size" question raised by the per-company
CUDA result.

Note: resolving across all companies together means entities are matched
globally (e.g. "Apple" mentioned in two different companies' filings would
resolve to the same node) rather than per-company. This is noted as a
methodological difference in the report, not a bug.
"""
from sentence_transformers import SentenceTransformer
import torch
import json
import time
import os

ENTITIES_PATH = "./data/entities.json"
OUTPUT_PATH = "./data/resolved_entities_cuda_batched.json"
SIMILARITY_THRESHOLD = 0.85


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on device: {device}")
    return SentenceTransformer("all-MiniLM-L6-v2", device=device), device


def cosine_sim_matrix_gpu(embeddings):
    norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    return norm @ norm.T


def resolve_entities_batched(model, entity_texts, device):
    unique_texts = list(set(entity_texts))
    n = len(unique_texts)
    print(f"Total unique entities across all companies: {n}")

    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
        batch_size=256,
    )

    sim_matrix = cosine_sim_matrix_gpu(embeddings)
    sim_matrix_cpu = sim_matrix.cpu().numpy()

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

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix_cpu[i][j] >= SIMILARITY_THRESHOLD:
                union(i, j)

    clusters = {}
    for i, text in enumerate(unique_texts):
        root = find(i)
        canonical = unique_texts[root]
        clusters.setdefault(canonical, []).append(text)

    mapping = {}
    for canonical, variants in clusters.items():
        for v in variants:
            mapping[v] = canonical

    return mapping


def main():
    model, device = load_model()

    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)

    # Flatten all entity texts across all companies into one list
    all_texts = []
    for company, entities in all_entities.items():
        all_texts.extend([e["text"] for e in entities])

    print(f"Running BATCHED CUDA entity resolution across {len(all_entities)} companies...")

    start = time.time()
    mapping = resolve_entities_batched(model, all_texts, device)
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    num_unique_resolved = len(set(mapping.values()))
    print(f"\nBatched CUDA entity resolution done. Total time: {elapsed:.2f}s")
    print(f"Device used: {device}")
    print(f"{len(set(all_texts))} unique mentions -> {num_unique_resolved} resolved entities")
    print(f"Results saved to {OUTPUT_PATH}")
    return elapsed


if __name__ == "__main__":
    main()
