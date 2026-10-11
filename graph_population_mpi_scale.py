"""
Stage 4 — MPI Graph Population scalability test.
Same logic as graph_population_mpi.py, but parameterized to accept
entities/relations file paths, so it can run on the 10/40/79-company
subsets for a corpus-size scaling table.

NOTE: does not use resolved_entities.json (per-company resolution) --
uses raw entity text directly, since the scale test subsets don't have
a matching resolved-entities file. This means canonicalization is
skipped here; fine for timing purposes, but worth noting in the report
that this run measures write throughput, not full pipeline fidelity.

Run with:
    mpirun -n 4 python graph_population_mpi_scale.py data/entities_10.json data/relations_10.json
"""
from mpi4py import MPI
from neo4j import GraphDatabase
import json
import time
import sys
import os

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "change-me")


class GraphWriter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def write_entity(self, tx, name, entity_type, company):
        tx.run(
            """
            MERGE (e:Entity {name: $name})
            SET e.type = $type
            MERGE (c:Company {name: $company})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            name=name, type=entity_type, company=company,
        )

    def write_relation(self, tx, subject, relation, obj, company):
        tx.run(
            """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            MERGE (s)-[r:RELATION {type: $relation}]->(o)
            SET r.source_company = $company
            """,
            subject=subject, relation=relation, object=obj, company=company,
        )

    def populate_company(self, company, entities, relations):
        with self.driver.session() as session:
            for e in entities:
                session.execute_write(self.write_entity, e["text"], e["label"], company)
            for r in relations:
                session.execute_write(self.write_relation, r["subject"], r["relation"], r["object"], company)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    entities_path = sys.argv[1]
    relations_path = sys.argv[2]

    if rank == 0:
        with open(entities_path, "r", encoding="utf-8") as f:
            all_entities = json.load(f)
        with open(relations_path, "r", encoding="utf-8") as f:
            all_relations = json.load(f)

        writer = GraphWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        with writer.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        writer.close()

        companies = sorted(all_entities.keys())
    else:
        all_entities = None
        all_relations = None
        companies = None

    all_entities = comm.bcast(all_entities, root=0)
    all_relations = comm.bcast(all_relations, root=0)
    companies = comm.bcast(companies, root=0)

    comm.Barrier()

    my_companies = companies[rank::size]

    if rank == 0:
        print(f"Running MPI graph population ({len(companies)} companies) with {size} process(es)...")

    writer = GraphWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    comm.Barrier()
    start = time.time()

    for company in my_companies:
        writer.populate_company(
            company,
            all_entities.get(company, []),
            all_relations.get(company, []),
        )

    writer.close()
    local_elapsed = time.time() - start

    all_timings = comm.gather(local_elapsed, root=0)

    comm.Barrier()
    if rank == 0:
        total_elapsed = max(all_timings)
        print(f"MPI graph population complete. Processes: {size} | "
              f"Companies: {len(companies)} | Total time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
