"""Ajustes no texto de documentos já gravados no grafo."""

import json
from typing import Callable, Dict, List

from extract_info.amendments import applyAmendments
from extract_info.glossary import expandAcronyms

queryNormTargets = """
MATCH (p:Document {id: $doc_id})-[:HAS_NORM]->(n:Document)
      -[:HAS_CAP]->(:Chapter)-[:HAS_SEC|HAS_CONT*1..2]->(r:Content)-[:REF_NORM]->(ct:Content)
RETURN ct.id AS id, coalesce(ct.texto_original, ct.texto) AS base,
       collect({fonte: n.titulo, texto: r.texto}) AS normas
"""

queryRewriteContent = """
MATCH (ct:Content {id: $id})
SET ct.texto_original = coalesce(ct.texto_original, $base),
    ct.texto = $texto,
    ct.alterado_por = $fontes
WITH ct
OPTIONAL MATCH (ct)-[:HAS_CHUNK]->(old:Chunk)
DETACH DELETE old
WITH DISTINCT ct
UNWIND $chunks AS chunk
MERGE (ch:Chunk {id: chunk.id})
SET ch:ContentChunk,
    ch.texto = chunk.texto,
    ch.embedding = chunk.embedding
MERGE (ct)-[:HAS_CHUNK]->(ch)
"""

querySaveAcronyms = """
MATCH (d:Document {id: $doc_id})
SET d.siglas = $siglas
"""

queryLoadAcronyms = """
MATCH (d:Document {id: $doc_id})
RETURN d.siglas AS siglas
"""


def saveAcronyms(db, docId: str, acronyms: Dict[str, str]):
    db.executeQuery(querySaveAcronyms, parameters={
        "doc_id": docId,
        "siglas": json.dumps(acronyms, ensure_ascii=False),
    })


def loadAcronyms(db, docId: str) -> Dict[str, str]:
    rows = db.executeQuery(queryLoadAcronyms, parameters={"doc_id": docId})
    raw = rows[0]["siglas"] if rows else None
    return json.loads(raw) if raw else {}


def consolidateNorms(db, docId: str, splitText: Callable[[str], List[str]],
                     embedTexts: Callable[[List[str]], List[List[float]]],
                     acronyms: Dict[str, str] = None) -> int:
    """Reescreve com o texto vigente os artigos alterados por resolução (original em texto_original)."""
    changed = 0

    for target in db.executeQuery(queryNormTargets, parameters={"doc_id": docId}):
        text = target["base"]
        sources = []

        for norm in target["normas"]:
            amended = applyAmendments(text, norm["texto"], norm["fonte"])
            if amended is None:
                print(f"  {norm['fonte']} não segue 'onde se lê / leia-se' para {target['id']}: mantido o texto original.")
                continue
            text = amended
            sources.append(norm["fonte"])

        if not sources:
            continue

        text = expandAcronyms(text, acronyms or {})
        pieces = [expandAcronyms(piece, acronyms or {}) for piece in splitText(text)]
        chunks = [
            {"id": f"{target['id']}_chunk{index}", "texto": piece, "embedding": vector}
            for index, (piece, vector) in enumerate(zip(pieces, embedTexts(pieces)))
        ]

        db.executeQuery(queryRewriteContent, parameters={
            "id": target["id"],
            "base": target["base"],
            "texto": text,
            "fontes": sources,
            "chunks": chunks,
        })
        changed += 1
        print(f"  {target['id']}: texto vigente aplicado ({', '.join(sources)}).")

    return changed

