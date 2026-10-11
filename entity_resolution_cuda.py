"""
Stage 3: Entity Resolution — CUDA-accelerated version.

Same logic as entity_resolution.py (sequential baseline), but:
  1. The embedding model runs on GPU (device='cuda')
  2. The pairwise cosine similarity matrix is computed as a GPU tensor
     operation (torch, CUDA backend) instead of NumPy on CPU

This is the stage where CUDA earns its place: computing similarity across
many entity pairs is repetitive, batchable math — ideal for GPU parallelism.
"""
from sentence_transformers import SentenceTransformer
import torch
import json
import time
import os

ENTITIES_PATH = "./data/entities.json"
OUTPUT_PATH = "./data/resolved_entities_cuda.json"
SIMILARITY_THRESHOLD = 0.85


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on device: {device}")
    return SentenceTransformer("all-MiniLM-L6-v2", device=device), device


def cosine_sim_matrix_gpu(embeddings, device):
    """
    Compute pairwise cosine similarity entirely on GPU.
    embeddings: torch tensor, shape (N, D), already on `device`.
    """
    norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    sim_matrix = norm @ norm.T   # GPU matrix multiply — this is the CUDA work
    return sim_matrix


def resolve_entities(model, entity_texts, device):
    unique_texts = list(set(entity_texts))
    if len(unique_texts) == 0:
        return {}

    # encode() with convert_to_tensor=True keeps embeddings on GPU,
    # avoiding an unnecessary GPU->CPU->GPU round trip
    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=False,
    )

    sim_matrix = cosine_sim_matrix_gpu(embeddings, device)
    # Pull the similarity matrix back to CPU only once, for the
    # union-find clustering step (that part is inherently sequential
    # bookkeeping, not worth parallelizing on GPU)
    sim_matrix_cpu = sim_matrix.cpu().numpy()

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

    print(f"Running CUDA entity resolution on {len(all_entities)} documents...")

    results = {}
    start = time.time()

    for i, (company, entities) in enumerate(all_entities.items(), 1):
        entity_texts = [e["text"] for e in entities]

        t0 = time.time()
        mapping = resolve_entities(model, entity_texts, device)
        t1 = time.time()

        results[company] = mapping
        num_unique = len(set(mapping.values()))
        print(f"[{i}/{len(all_entities)}] {company}: "
              f"{len(entity_texts)} mentions -> {num_unique} resolved entities ({t1 - t0:.3f}s)")

    elapsed = time.time() - start
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nCUDA entity resolution done. Total time: {elapsed:.2f}s")
    print(f"Device used: {device}")
    print(f"Results saved to {OUTPUT_PATH}")
    return elapsed


if __name__ == "__main__":
    main()
