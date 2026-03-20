from connection import Neo4jConnection, URI, USER, PASSWORD
import os
DATABASE = "rag_documents"
db = Neo4jConnection(URI, USER, PASSWORD)

query_constraints_document = """
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document)
REQUIRE (d.id, d.titulo) IS UNIQUE;
"""

query_constraints_chapter = """
CREATE CONSTRAINT chapter_unique IF NOT EXISTS
FOR (c:Chapter)
REQUIRE c.id IS UNIQUE;
"""

query_constraints_section = """
CREATE CONSTRAINT section_unique IF NOT EXISTS
FOR (s:Section)
REQUIRE s.id IS UNIQUE;
"""
query_constraints_content = """
CREATE CONSTRAINT content_unique IF NOT EXISTS
FOR (ct:Content)
REQUIRE ct.id IS UNIQUE;
"""

query_constraints_embedding = """
CREATE CONSTRAINT embedding_exists IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.id IS UNIQUE;
"""

query_index = """
CREATE VECTOR INDEX content_embedding IF NOT EXISTS
FOR (c:Chunk)
ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }
};
"""

query_constraints_user = """
CREATE CONSTRAINT user_unique IF NOT EXISTS
FOR (u:User)
REQUIRE u.email IS UNIQUE;
"""


query_delete_all = """
MATCH (n)
DETACH DELETE n;
"""

query_delete_all_constraints = """
SHOW CONSTRAINTS YIELD name
RETURN "DROP CONSTRAINT " + name + ";"
"""

query_delete_all_indexes = """
SHOW INDEXES YIELD name
RETURN "DROP INDEX " + name + ";"
"""

if __name__ == "__main__":
    db.execute_query(query_delete_all)
    db.execute_query(query_delete_all_constraints)
    db.execute_query(query_delete_all_indexes)
    
    db.execute_query(query_constraints_document)
    print("Document constraints created successfully.")
    db.execute_query(query_constraints_chapter)
    print("Chapter constraints created successfully.")
    db.execute_query(query_constraints_section)
    print("Section constraints created successfully.")
    db.execute_query(query_constraints_content)
    print("Content constraints created successfully.")
    db.execute_query(query_constraints_embedding)
    print("Embedding constraints created successfully.")
    db.execute_query(query_index)
    print("Vector index created successfully.")
    
    db.execute_query(query_constraints_user)