

def dict_to_list(doc, embed_func: callable):
    capitulos_lista = []
    refs = []
    for cap_id, cap in doc["capitulos"].items():
        secoes_lista = []
        conteudos_cap_lista = []

        for sec_id, sec in cap["secoes"].items():
            conteudos_lista = []

            for cont_id, cont in sec["conteudos"].items():
                chunk_lista = []
                for chunk in cont["chunks"]:
                    chunk_lista.append({
                        "id": chunk["id"],
                        "embedding": embed_func(chunk["texto"])
                    })
                refs.append({
                    "origem": cont["id"],
                    "destino": cont.get("refs", [])
                    })
                
                conteudos_lista.append({
                    "id": cont["id"],
                    "num": cont["cont_num"],
                    "tipo": cont["tipo"],
                    "texto": cont["texto"],
                    "chunks": chunk_lista,
                })

            secoes_lista.append({
                "id": sec["id"],
                "secao": sec["secao"],
                "conteudos": conteudos_lista
            })
        
        for cont_id, cont in cap["conteudos"].items():
            chunk_lista = []
            for chunk in cont["chunks"]:
                chunk_lista.append({
                    "id": chunk["id"],
                    "embedding": embed_func(chunk["texto"])
                })
            refs.append({
                "origem": cont["id"],
                "destino": cont.get("refs", [])
            })
            conteudos_cap_lista.append({
                "id": cont["id"],
                "tipo": cont["tipo"],
                "num": cont["cont_num"],
                "texto": cont["texto"],
                "chunks": chunk_lista,
            })
        
        capitulos_lista.append({
            "id": cap["id"],
            "capitulo": cap["capitulo"],
            "secoes": secoes_lista,
            "conteudos": conteudos_cap_lista
        })
    
    return {
        "id": doc["id"],
        "titulo": doc.get("titulo", ""),
        "path": doc.get("path", ""),
        "capitulos": capitulos_lista,
        "refs": refs
    }
    
def inserir_estrutura(tx, doc):
    
    
    query_cap = """
MERGE (d:Document {id: $doc_id})
SET d.titulo = $doc_titulo,
    d.path = $doc_path

WITH d
UNWIND $capitulos AS cap

MERGE (c:Chapter {id: cap.id})
SET c.capitulo = cap.capitulo,
    c.documento_id = $doc_id

MERGE (d)-[:HAS_CAP]->(c)
"""

    query_cont_cap = """
UNWIND $capitulos AS cap

MATCH (c:Chapter {id: cap.id})

UNWIND coalesce(cap.conteudos, []) AS cont

MERGE (ct:Content {id: cont.id})
SET ct.tipo = cont.tipo,
    ct.num = cont.num,
    ct.texto = cont.texto,
    ct.documento_id = $doc_id

MERGE (c)-[:HAS_CONT]->(ct)


"""

    query_cont_sec = """
    UNWIND $capitulos AS cap
MATCH (c:Chapter {id: cap.id})

UNWIND coalesce(cap.secoes, []) AS sec

MERGE (s:Section {id: sec.id})
SET s.secao = sec.secao,
    s.capitulo_id = cap.id

MERGE (c)-[:HAS_SEC]->(s)

WITH sec, s
UNWIND coalesce(sec.conteudos, []) AS cont

MERGE (ct:Content {id: cont.id})
SET ct.tipo = cont.tipo,
    ct.num = cont.num,
    ct.texto = cont.texto,
    ct.documento_id = $doc_id

MERGE (s)-[:HAS_CONT]->(ct)"""

    query_chunk_cap = """
UNWIND $capitulos AS cap

UNWIND coalesce(cap.conteudos, []) AS cont
MATCH (ct:Content {id: cont.id})

UNWIND coalesce(cont.chunks, []) AS chunk

MERGE (ch:Chunk {id: chunk.id})
SET ch.texto = chunk.texto,
    ch.embedding = chunk.embedding

MERGE (ct)-[:HAS_CHUNK]->(ch)

"""

    query_chunk_sec = """
    UNWIND $capitulos AS cap
    UNWIND coalesce(cap.secoes, []) AS sec
UNWIND coalesce(sec.conteudos, []) AS cont

MATCH (ct:Content {id: cont.id})

UNWIND coalesce(cont.chunks, []) AS chunk

MERGE (ch:Chunk {id: chunk.id})
SET ch.texto = chunk.texto,
    ch.embedding = chunk.embedding

MERGE (ct)-[:HAS_CHUNK]->(ch)
"""

    query_refs = """
UNWIND $refs AS ref

MATCH (orig:Content {id: ref.origem})

UNWIND ref.destino AS num_ref

MATCH (dest:Content {
    num: num_ref,
    documento_id: $doc_id
})

WITH orig, dest
WHERE dest IS NOT NULL

MERGE (orig)-[:REFERENCES]->(dest)
"""
    # tx.run(query,
    #        doc_id=doc["id"],
    #        doc_titulo=doc.get("titulo", ""),
    #        doc_path=doc["path"],
    #        capitulos=doc["capitulos"])
    tx.execute_query(query_cap,
           parameters={
               "doc_id": doc["id"],
               "doc_titulo": doc.get("titulo", ""),
               "doc_path": doc["path"],
               "capitulos": doc["capitulos"]
           })
    tx.execute_query(query_cont_cap,
           parameters={
               "capitulos": doc["capitulos"],
               "doc_id": doc["id"]
           })
    tx.execute_query(query_cont_sec,
           parameters={
               "capitulos": doc["capitulos"],
               "doc_id": doc["id"]
           })
    tx.execute_query(query_refs,
           parameters={
               "refs": doc["refs"],
               "doc_id": doc["id"]
           })
    tx.execute_query(query_chunk_cap,
           parameters={
               "capitulos": doc["capitulos"]
           })
    tx.execute_query(query_chunk_sec,
           parameters={
               "capitulos": doc["capitulos"]
           })
    
    
def retrieve_all_documents(tx):
    query = """
    MATCH (d:Document)
    RETURN d.id AS id, d.titulo AS titulo, d.path AS path
    """
    result = tx.run(query)
    return [record.data() for record in result]

def delete_document(tx, doc_id):
    query = """
    MATCH (d:Documento {id: $doc_id})
    CALL {
        WITH d
        MATCH (d)-[*]->(sub)
        RETURN collect(DISTINCT sub) AS nodes
    }
    WITH d, nodes
    FOREACH (n IN nodes | DETACH DELETE n)
    DETACH DELETE d
    """
    tx.run(query, doc_id=doc_id)