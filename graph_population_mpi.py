"""
Stage 4: Graph Population — MPI-parallelized version.

Uses mpi4py (real MPI, via your installed MPICH) to split the 79 companies
across N processes. Each process independently writes its share of
companies' entities/relations into Neo4j, running concurrently.

This is the stage most analogous to true distributed computing: each
MPI rank is an independent process (could be on a different machine
entirely), coordinating only via message passing (here, mostly implicit:
each rank works on a disjoint slice of companies, so there's minimal need
for processes to talk to each other, aside from the final gather of timings).

Run with:
    mpirun -n 4 python graph_population_mpi.py

(-n 4 = number of MPI processes; try 2, 4, 8 to build a scaling curve,
same as we did for OpenMP)
"""
from mpi4py import MPI
from neo4j import GraphDatabase
import json
import time

ENTITIES_PATH = "./data/entities.json"
RELATIONS_PATH = "./data/relations.json"
RESOLVED_PATH = "./data/resolved_entities.json"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "os.environ.get("NEO4J_PASSWORD", "change-me") 


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

    def populate_company(self, company, entities, relations, resolved_map):
        with self.driver.session() as session:
            for e in entities:
                canonical = resolved_map.get(e["text"], e["text"])
                session.execute_write(self.write_entity, canonical, e["label"], company)
            for r in relations:
                subj = resolved_map.get(r["subject"], r["subject"])
                obj = resolved_map.get(r["object"], r["object"])
                session.execute_write(self.write_relation, subj, r["relation"], obj, company)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()      # this process's ID (0, 1, 2, ...)
    size = comm.Get_size()      # total number of MPI processes

    # Only rank 0 clears the database first, and all ranks wait for it
    # to finish before starting writes (via comm.Barrier — a real MPI
    # synchronization primitive: every process pauses here until all
    # processes have reached this point)
    if rank == 0:
        with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
            all_entities = json.load(f)
        with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
            all_relations = json.load(f)
        with open(RESOLVED_PATH, "r", encoding="utf-8") as f:
            all_resolved = json.load(f)

        print(f"[Rank 0] Clearing existing graph...")
        writer = GraphWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        with writer.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        writer.close()

        companies = sorted(all_entities.keys())
    else:
        all_entities = None
        all_relations = None
        all_resolved = None
        companies = None

    # Broadcast the data to all processes (MPI's collective communication:
    # rank 0 loaded the JSON files, now every process gets a copy)
    all_entities = comm.bcast(all_entities, root=0)
    all_relations = comm.bcast(all_relations, root=0)
    all_resolved = comm.bcast(all_resolved, root=0)
    companies = comm.bcast(companies, root=0)

    comm.Barrier()  # wait until rank 0's clear-database step is fully done

    # Each rank gets a disjoint slice of the company list
    my_companies = companies[rank::size]  # stride-based split across ranks

    if rank == 0:
        print(f"Running MPI graph population with {size} process(es)...")

    writer = GraphWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    comm.Barrier()  # all ranks start timing at the same moment
    start = time.time()

    for company in my_companies:
        writer.populate_company(
            company,
            all_entities.get(company, []),
            all_relations.get(company, []),
            all_resolved.get(company, {}),
        )
        print(f"[Rank {rank}] {company}: written")

    writer.close()
    local_elapsed = time.time() - start

    # Gather all ranks' timings back to rank 0 for reporting
    all_timings = comm.gather(local_elapsed, root=0)

    comm.Barrier()
    if rank == 0:
        total_elapsed = max(all_timings)  # wall-clock time = slowest rank
        print(f"\nMPI graph population complete.")
        print(f"Processes used: {size}")
        print(f"Per-rank times: {[f'{t:.2f}s' for t in all_timings]}")
        print(f"Total wall-clock time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
