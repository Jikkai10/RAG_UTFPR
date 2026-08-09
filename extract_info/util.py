

def _contentToDict(content, embedFunc: callable):
    chunkList = [
        {
            "id": chunk["id"],
            "texto": chunk["texto"],
            "embedding": embedFunc(chunk["texto"]),
        }
        for chunk in content["chunks"]
    ]

    return {
        "id": content["id"],
        "num": content["cont_num"],
        "tipo": content["tipo"],
        "texto": content["texto"],
        "pagina_inicio": content.get("pagina_inicio"),
        "pagina_fim": content.get("pagina_fim"),
        "bbox": content.get("bbox"),
        "chunks": chunkList,
    }


def dictToList(doc, embedFunc: callable):
    chapterList = []
    refs = []
    for chapterId, chapter in doc["capitulos"].items():
        sectionList = []
        chapterContentList = []

        for sectionId, section in chapter["secoes"].items():
            contentList = []

            for contentId, content in section["conteudos"].items():
                refs.append({
                    "origem": content["id"],
                    "destino": content.get("refs", [])
                })

                contentList.append(_contentToDict(content, embedFunc))

            sectionList.append({
                "id": section["id"],
                "secao": section["secao"],
                "conteudos": contentList
            })

        for contentId, content in chapter["conteudos"].items():
            refs.append({
                "origem": content["id"],
                "destino": content.get("refs", [])
            })
            chapterContentList.append(_contentToDict(content, embedFunc))

        chapterList.append({
            "id": chapter["id"],
            "capitulo": chapter["capitulo"],
            "secoes": sectionList,
            "conteudos": chapterContentList
        })

    return {
        "id": doc["id"],
        "titulo": doc.get("titulo", ""),
        "path": doc.get("path", ""),
        "capitulos": chapterList,
        "refs": refs
    }

def insertCalendar(tx, doc):

    queryReset = """
MATCH (d:Document {id: $doc_id})-[:HAS_EVENT]->(ev:Events)

OPTIONAL MATCH (ev)-[:HAS_ITEM]->(it:EventItem)
OPTIONAL MATCH (ev)-[:HAS_CHUNK]->(chs:Chunk)
OPTIONAL MATCH (it)-[:HAS_CHUNK]->(chi:Chunk)

DETACH DELETE chi, chs, it, ev
"""

    queryDoc = """
MERGE (d:Document {id: $doc_id})
SET d.titulo = $doc_titulo,
    d.path = $doc_path,
    d.tipo = 3
"""

    querySections = """
MATCH (d:Document {id: $doc_id})
UNWIND $secoes AS secao

MERGE (ev:Events {id: secao.id})
SET ev.campus = secao.campus,
    ev.categoria = secao.categoria,
    ev.periodo = secao.periodo,
    ev.ano = secao.ano,
    ev.pagina = secao.pagina,
    ev.tipo = secao.tipo,
    ev.texto = secao.texto

MERGE (d)-[:HAS_EVENT]->(ev)
"""

    querySectionChunks = """
UNWIND $secoes AS secao
MATCH (ev:Events {id: secao.id})

UNWIND coalesce(secao.chunks, []) AS chunk

MERGE (ch:Chunk {id: chunk.id})
SET ch:EventChunk,
    ch.texto = chunk.texto,
    ch.embedding = chunk.embedding

MERGE (ev)-[:HAS_CHUNK]->(ch)
"""

    queryItems = """
UNWIND $secoes AS secao
MATCH (ev:Events {id: secao.id})

UNWIND coalesce(secao.itens, []) AS item

MERGE (it:EventItem {id: item.id})
SET it.descricao = item.descricao,
    it.dia_texto = item.dia_texto,
    it.mes = item.mes,
    it.mes_num = item.mes_num,
    it.ano = item.ano,
    it.periodo = item.periodo,
    it.texto = item.texto,
    it.data_inicio = CASE WHEN item.data_inicio IS NULL THEN NULL ELSE date(item.data_inicio) END,
    it.data_fim = CASE WHEN item.data_fim IS NULL THEN NULL ELSE date(item.data_fim) END

MERGE (ev)-[:HAS_ITEM]->(it)

WITH it, item
UNWIND coalesce(item.chunks, []) AS chunk

MERGE (ch:Chunk {id: chunk.id})
SET ch:EventChunk,
    ch.texto = chunk.texto,
    ch.embedding = chunk.embedding

MERGE (it)-[:HAS_CHUNK]->(ch)
"""

    parameters = {
        "doc_id": doc["doc_id"],
        "doc_titulo": doc.get("titulo", ""),
        "doc_path": doc["path"],
        "secoes": doc["secoes"],
    }

    for query in (queryReset, queryDoc, querySections, querySectionChunks, queryItems):
        tx.executeQuery(query, parameters=parameters)


def insertStructure(tx, doc, docType, docId = None):


    queryChapter = """
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

    queryNorm = """
        MATCH (d:Document {id: $doc_id})
        MATCH (p:Document {id: $pai_id})

        MERGE (p)-[:HAS_NORM]->(d)
    """

    queryResetChunks = """
MATCH (d:Document {id: $doc_id})-[:HAS_CAP]->(:Chapter)
      -[:HAS_SEC|HAS_CONT*1..2]->(:Content)-[:HAS_CHUNK]->(ch:Chunk)
DETACH DELETE ch
"""


    queryChapterContent = """
UNWIND $capitulos AS cap

MATCH (c:Chapter {id: cap.id})

UNWIND coalesce(cap.conteudos, []) AS cont

MERGE (ct:Content {id: cont.id})
SET ct.tipo = cont.tipo,
    ct.num = cont.num,
    ct.texto = cont.texto,
    ct.pagina_inicio = cont.pagina_inicio,
    ct.pagina_fim = cont.pagina_fim,
    ct.bbox = cont.bbox,
    ct.documento_id = $doc_id

MERGE (c)-[:HAS_CONT]->(ct)


"""

    querySectionContent = """
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
    ct.pagina_inicio = cont.pagina_inicio,
    ct.pagina_fim = cont.pagina_fim,
    ct.bbox = cont.bbox,
    ct.documento_id = $doc_id

MERGE (s)-[:HAS_CONT]->(ct)"""

    queryChapterChunk = """
UNWIND $capitulos AS cap

UNWIND coalesce(cap.conteudos, []) AS cont
MATCH (ct:Content {id: cont.id})

UNWIND coalesce(cont.chunks, []) AS chunk

MERGE (ch:Chunk {id: chunk.id})
SET ch:ContentChunk,
    ch.texto = chunk.texto,
    ch.embedding = chunk.embedding

MERGE (ct)-[:HAS_CHUNK]->(ch)

"""

    querySectionChunk = """
    UNWIND $capitulos AS cap
    UNWIND coalesce(cap.secoes, []) AS sec
UNWIND coalesce(sec.conteudos, []) AS cont

MATCH (ct:Content {id: cont.id})

UNWIND coalesce(cont.chunks, []) AS chunk

MERGE (ch:Chunk {id: chunk.id})
SET ch:ContentChunk,
    ch.texto = chunk.texto,
    ch.embedding = chunk.embedding

MERGE (ct)-[:HAS_CHUNK]->(ch)
"""

    queryRefs = """
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

    queryRefsNorm = """
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
    tx.executeQuery(queryResetChunks,
           parameters={
               "doc_id": doc["id"]
           })
    tx.executeQuery(queryChapter,
           parameters={
               "doc_id": doc["id"],
               "doc_titulo": doc.get("titulo", ""),
               "doc_type": docType,
               "doc_path": doc["path"],
               "capitulos": doc["capitulos"]
           })
    tx.executeQuery(queryChapterContent,
           parameters={
               "capitulos": doc["capitulos"],
               "doc_id": doc["id"]
           })
    tx.executeQuery(querySectionContent,
           parameters={
               "capitulos": doc["capitulos"],
               "doc_id": doc["id"]
           })

    tx.executeQuery(queryChapterChunk,
        parameters={
            "capitulos": doc["capitulos"]
        })
    tx.executeQuery(querySectionChunk,
        parameters={
            "capitulos": doc["capitulos"]
        })

    if docType == 0:
        tx.executeQuery(queryRefs,
            parameters={
                "refs": doc["refs"],
                "doc_id": doc["id"]
            })
    else:
        print(doc["refs"])
        tx.executeQuery(queryRefsNorm,
            parameters={
                "refs": doc["refs"],
                "doc_id": docId
            })
        tx.executeQuery(queryNorm,
            parameters={
                "doc_id": doc["id"],
                "pai_id": docId
            })


def retrieveAllDocuments(tx):
    query = """
    MATCH (d:Document)
    WHERE d.tipo <> $tipo

    OPTIONAL MATCH (d)-[:HAS_NORM]->(n:Document)

    RETURN d, collect(n) AS norms
    """

    result = tx.executeQuery(query, parameters = {"tipo": 1})

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

        node = record["d"]

        doc = {
            "id": node["id"],
            "titulo": node["titulo"],
            "path": node["path"],
            "norms": norms
        }
        docs.append(doc)

    return docs

def returnDocument(tx, docId):
    query= """
    MATCH (d:Document {id: $doc_id})

    RETURN d
    """
    result = tx.executeQuery(query, parameters = {"doc_id": docId})
    record = result[0]
    return record["d"]

def deleteDocument(tx, docId):
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
    result = tx.executeQuery(query, parameters = {"doc_id": docId})
    paths = []
    record = result[0]
    paths.append(record["doc_path"])
    for norm in record["norm_paths"]:
        if norm:
            paths.append(norm)
    print(paths)
    return paths
