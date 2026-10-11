"""
Runs the full sequential baseline pipeline end-to-end and records
per-stage timing. This is the number every parallel version gets
compared against.

Usage:
    python run_baseline.py            # full pipeline (needs Neo4j running)
    python run_baseline.py --no-graph # skip Neo4j step (NER/RE/ER only)
"""
import time
import json
import sys
import os

import ner_sequential
import relation_extraction
import entity_resolution

TIMING_OUTPUT = "./data/baseline_timings.json"


def main():
    skip_graph = "--no-graph" in sys.argv
    timings = {}

    print("=" * 60)
    print("STAGE 1: Named Entity Recognition")
    print("=" * 60)
    timings["ner"] = ner_sequential.main()

    print("\n" + "=" * 60)
    print("STAGE 2: Relation Extraction")
    print("=" * 60)
    timings["relation_extraction"] = relation_extraction.main()

    print("\n" + "=" * 60)
    print("STAGE 3: Entity Resolution")
    print("=" * 60)
    timings["entity_resolution"] = entity_resolution.main()

    if not skip_graph:
        print("\n" + "=" * 60)
        print("STAGE 4: Graph Population")
        print("=" * 60)
        import graph_population
        timings["graph_population"] = graph_population.main()
    else:
        print("\nSkipping graph population stage (--no-graph flag set)")

    timings["total"] = sum(v for v in timings.values() if v is not None)

    os.makedirs(os.path.dirname(TIMING_OUTPUT), exist_ok=True)
    with open(TIMING_OUTPUT, "w") as f:
        json.dump(timings, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)
    for stage, t in timings.items():
        print(f"  {stage}: {t:.2f}s")
    print(f"\nTimings saved to {TIMING_OUTPUT}")


if __name__ == "__main__":
    main()
