import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import Neo4jConnection, URI, USER, PASSWORD
from db.create_users import createDefaultUsers

DATABASE = "rag_documents"

queryConstraintsDocument = """
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document)
REQUIRE (d.id, d.titulo) IS UNIQUE;
"""

queryConstraintsChapter = """
CREATE CONSTRAINT chapter_unique IF NOT EXISTS
FOR (c:Chapter)
REQUIRE c.id IS UNIQUE;
"""

queryConstraintsSection = """
CREATE CONSTRAINT section_unique IF NOT EXISTS
FOR (s:Section)
REQUIRE s.id IS UNIQUE;
"""
queryConstraintsContent = """
CREATE CONSTRAINT content_unique IF NOT EXISTS
FOR (ct:Content)
REQUIRE ct.id IS UNIQUE;
"""

queryConstraintsEmbedding = """
CREATE CONSTRAINT embedding_exists IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.id IS UNIQUE;
"""

queryIndexContent = """
CREATE VECTOR INDEX content_chunk_embedding IF NOT EXISTS
FOR (c:ContentChunk)
ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};
"""

queryIndexEvent = """
CREATE VECTOR INDEX event_chunk_embedding IF NOT EXISTS
FOR (c:EventChunk)
ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};
"""

queryLabelContentChunks = """
MATCH (:Content)-[:HAS_CHUNK]->(ch:Chunk)
WHERE NOT ch:ContentChunk
SET ch:ContentChunk
"""

queryLabelEventChunks = """
MATCH (:Events)-[:HAS_CHUNK]->(ch:Chunk)
WHERE NOT ch:EventChunk
SET ch:EventChunk
"""

queryDropLegacyIndex = """
DROP INDEX content_embedding IF EXISTS
"""

queryConstraintsUser = """
CREATE CONSTRAINT user_unique IF NOT EXISTS
FOR (u:User)
REQUIRE u.email IS UNIQUE;
"""

queryConstraintsEvents = """
CREATE CONSTRAINT events_unique IF NOT EXISTS
FOR (ev:Events)
REQUIRE ev.id IS UNIQUE;
"""

queryConstraintsEventItem = """
CREATE CONSTRAINT event_item_unique IF NOT EXISTS
FOR (it:EventItem)
REQUIRE it.id IS UNIQUE;
"""

queryIndexEventDate = """
CREATE INDEX event_item_data IF NOT EXISTS
FOR (it:EventItem)
ON (it.data_inicio);
"""

queryPurgeCalendars = """
MATCH (d:Document {tipo: 3})

OPTIONAL MATCH (d)-[:HAS_EVENT]->(ev:Events)
OPTIONAL MATCH (ev)-[:HAS_ITEM]->(it:EventItem)
OPTIONAL MATCH (ev)-[:HAS_CHUNK]->(chs:Chunk)
OPTIONAL MATCH (it)-[:HAS_CHUNK]->(chi:Chunk)

DETACH DELETE chi, chs, it, ev, d
"""

queryPurgeOrphans = """
MATCH (ev:Events)
WHERE NOT (:Document)-[:HAS_EVENT]->(ev)

OPTIONAL MATCH (ev)-[:HAS_ITEM]->(it:EventItem)
OPTIONAL MATCH (ev)-[:HAS_CHUNK]->(chs:Chunk)
OPTIONAL MATCH (it)-[:HAS_CHUNK]->(chi:Chunk)

DETACH DELETE chi, chs, it, ev
"""


queryDeleteAll = """
MATCH (n)
DETACH DELETE n;
"""

queryListConstraints = """
SHOW CONSTRAINTS YIELD name
RETURN name
"""

queryListIndexes = """
SHOW INDEXES YIELD name, type
WHERE type <> "LOOKUP"
RETURN name
"""

SCHEMA_QUERIES = [
    (queryConstraintsDocument, "Document constraints"),
    (queryConstraintsChapter, "Chapter constraints"),
    (queryConstraintsSection, "Section constraints"),
    (queryConstraintsContent, "Content constraints"),
    (queryConstraintsEmbedding, "Embedding constraints"),
    (queryLabelContentChunks, "Content chunk labels"),
    (queryLabelEventChunks, "Event chunk labels"),
    (queryDropLegacyIndex, "Legacy shared vector index dropped if present"),
    (queryIndexContent, "Content vector index"),
    (queryIndexEvent, "Event vector index"),
    (queryConstraintsEvents, "Events constraints"),
    (queryConstraintsEventItem, "EventItem constraints"),
    (queryIndexEventDate, "EventItem date index"),
    (queryConstraintsUser, "User constraints"),
]


def reset(db):
    db.executeQuery(queryDeleteAll)
    print("All nodes deleted.")

    for record in db.executeQuery(queryListConstraints):
        db.executeQuery(f"DROP CONSTRAINT {record['name']}")
    print("All constraints dropped.")

    for record in db.executeQuery(queryListIndexes):
        db.executeQuery(f"DROP INDEX {record['name']}")
    print("All indexes dropped.")


def purgeCalendars(db):
    sectionsBefore = db.executeQuery(
        "MATCH (ev:Events) RETURN count(ev) AS secoes"
    )[0]["secoes"]

    db.executeQuery(queryPurgeCalendars)
    db.executeQuery(queryPurgeOrphans)

    print(f"{sectionsBefore} secoes de calendario apagadas. Reenvie os PDFs de calendario.")


def bootstrap(db=None, withUsers=True):
    db = db or Neo4jConnection(URI, USER, PASSWORD)

    for query, label in SCHEMA_QUERIES:
        db.executeQuery(query)
        print(f"{label}: ok.")

    if withUsers:
        createDefaultUsers(db)

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicializa o schema do Neo4j.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="APAGA todos os nos, constraints e indexes antes de criar o schema.",
    )
    parser.add_argument(
        "--no-users",
        dest="noUsers",
        action="store_true",
        help="Nao cria os usuarios padrao.",
    )
    parser.add_argument(
        "--purge-calendars",
        dest="purgeCalendars",
        action="store_true",
        help="APAGA os calendarios (Events, EventItem e seus chunks) e os Documents tipo 3. "
             "Os regulamentos ficam. Use uma vez para limpar as secoes gravadas com o id "
             "antigo, e reenvie os PDFs de calendario depois.",
    )
    args = parser.parse_args()

    db = Neo4jConnection(URI, USER, PASSWORD)

    if args.reset:
        reset(db)

    if args.purgeCalendars:
        purgeCalendars(db)

    bootstrap(db, withUsers=not args.noUsers)
