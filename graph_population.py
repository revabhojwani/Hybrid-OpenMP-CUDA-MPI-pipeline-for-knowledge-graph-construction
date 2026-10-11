"""
Stage 4: Graph Population — SEQUENTIAL baseline.
Writes resolved entities and relations into Neo4j, one company at a time,
one triple at a time. This is the stage MPI will later parallelize
(splitting companies across multiple processes writing concurrently).
"""
from neo4j import GraphDatabase
import json
import time

ENTITIES_PATH = "./data/entities.json"
RELATIONS_PATH = "./data/relations.json"
RESOLVED_PATH = "./data/resolved_entities.json"

# Update these to match your local Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "os.environ.get("NEO4J_PASSWORD", "change-me")"


class GraphWriter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

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
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        all_entities = json.load(f)
    with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
        all_relations = json.load(f)
    with open(RESOLVED_PATH, "r", encoding="utf-8") as f:
        all_resolved = json.load(f)

    writer = GraphWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print("Clearing existing graph...")
    writer.clear_database()

    print(f"Populating graph for {len(all_entities)} companies (sequential)...")
    start = time.time()

    for i, company in enumerate(all_entities.keys(), 1):
        t0 = time.time()
        writer.populate_company(
            company,
            all_entities.get(company, []),
            all_relations.get(company, []),
            all_resolved.get(company, {}),
        )
        t1 = time.time()
        print(f"[{i}/{len(all_entities)}] {company}: written ({t1 - t0:.2f}s)")

    elapsed = time.time() - start
    writer.close()

    print(f"\nSequential graph population done. Total time: {elapsed:.2f}s")
    return elapsed


if __name__ == "__main__":
    main()
