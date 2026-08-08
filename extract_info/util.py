

def dict_to_list(doc, doc_type, embed_func: callable):
    capitulos_lista = []
    refs = []
    for cap_id, cap in doc["capitulos"].items():
        secoes_lista = []
        conteudos_cap_lista = []

        for sec_id, sec in cap["secoes"].items():
            conteudos_lista = []

            for cont_id, cont in sec["conteudos"].items():
                chunk_lista = []
                if doc_type == 0:
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
    
def insert_calendar(tx, doc):
    query = """
        MERGE (d:Document {id: $doc_id})
        SET d.titulo = $doc_titulo,
            d.path = $doc_path,
            d.tipo = 3

        WITH d
        UNWIND $events AS event

        MERGE (e:Events {id: event.id})
        SET e.campus = event.campus,
            e.categoria = event.categoria,
            e.periodo = event.periodo,
            e.texto = event.md

        MERGE (d)-[:HAS_EVENT]->(e)

        WITH e, event
        UNWIND coalesce(event.chunks, []) AS chunk

        MERGE (ch:Chunk {id: chunk.id})
        SET ch.texto = chunk.texto,
            ch.embedding = chunk.embedding

        MERGE (e)-[:HAS_CHUNK]->(ch)
    """
    
    tx.execute_query(query,
           parameters={
               "doc_id": doc["doc_id"],
               "doc_titulo": doc.get("titulo", ""),
               "doc_path": doc["path"],
               "events": doc["parts"]
           })
    
    
    
def inserir_estrutura(tx, doc, doc_type, doc_id = None):
    
    
    query_cap = """
MERGE (d:Document {id: $doc_id})
SET d.titulo = $doc_titulo,
    d.path = $doc_path,
    d.tipo = $doc_type

WITH d
UNWIND $capitulos AS cap

MERGE (c:Chapter {id: cap.id})
SET c.capitulo = cap.capitulo,
    c.documento_id = $doc_id

MERGE (d)-[:HAS_CAP]->(c)
"""

    query_norm = """
        MATCH (d:Document {id: $doc_id})
        MATCH (p:Document {id: $pai_id})
        
        MERGE (p)-[:HAS_NORM]->(d)
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

    query_refs_norm = """
UNWIND $refs AS ref

MATCH (orig:Content {id: ref.origem})

UNWIND ref.destino AS num_ref

MATCH (dest:Content {
    num: num_ref,
    documento_id: $doc_id
})

WITH orig, dest
WHERE dest IS NOT NULL

MERGE (orig)-[:REF_NORM]->(dest)
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
               "doc_type": doc_type,
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
    if doc_type == 0:
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
    else:
        print(doc["refs"])
        tx.execute_query(query_refs_norm,
            parameters={
                "refs": doc["refs"],
                "doc_id": doc_id
            })
        tx.execute_query(query_norm,
            parameters={
                "doc_id": doc["id"],
                "pai_id": doc_id
            })
    
    
def retrieve_all_documents(tx):
    query = """
    MATCH (d:Document)
    WHERE d.tipo <> $tipo

    OPTIONAL MATCH (d)-[:HAS_NORM]->(n:Document)

    RETURN d, collect(n) AS norms
    """

    result = tx.execute_query(query, parameters = {"tipo": 1})

    docs = []
    for record in result:
        norms = []
        for norm in record["norms"]:
            norms.append(
                {
                    "id": norm["id"],
                    "titulo": norm["titulo"],
                    "path": norm["path"]
                }
            )
        
        d = record["d"]
        
        doc = {
            "id": d["id"],
            "titulo": d["titulo"],
            "path": d["path"],
            "norms": norms
        }
        docs.append(doc)
    
    return docs

def return_document(tx, doc_id):
    query= """
    MATCH (d:Document {id: $doc_id})
    
    RETURN d
    """
    result = tx.execute_query(query, parameters = {"doc_id": doc_id})
    record = result[0]
    return record["d"]

def delete_document(tx, doc_id):
    query = """
    MATCH (d:Document {id: $doc_id})

    OPTIONAL MATCH (d)-[:HAS_NORM]->(n)
    WITH d, d.path AS doc_path, collect(DISTINCT n.path) AS norm_paths

    OPTIONAL MATCH (d)-[:HAS_NORM|HAS_CAP|HAS_SEC|HAS_CONT|HAS_CHUNK*1..5]->(sub)
    WITH doc_path, norm_paths, collect(DISTINCT sub) + collect(d) AS nodes

    UNWIND nodes AS node
    DETACH DELETE node

    RETURN doc_path, norm_paths
    """
    result = tx.execute_query(query, parameters = {"doc_id": doc_id})
    path = []
    record = result[0]
    path.append(record["doc_path"])
    for norm in record["norm_paths"]:
        if norm:
            path.append(norm)
    print(path)
    return path