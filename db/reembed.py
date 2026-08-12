"""Regera os embeddings dos Chunks ja gravados no grafo.

Necessario sempre que o modelo de embedding ou os prefixos do e5 mudarem: os
vetores antigos foram gerados com outra convencao e deixam de ser comparaveis
com o vetor da pergunta. Le o texto que ja esta no grafo, entao nao reprocessa
PDF nem chama o LLM.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL, buildEmbeddings
from db.connection import Neo4jConnection

BATCH_SIZE = 32

queryFetchChunks = """
MATCH (ch:Chunk)
WHERE ch.texto IS NOT NULL
RETURN ch.id AS id, ch.texto AS texto
ORDER BY ch.id
"""

queryUpdateEmbeddings = """
UNWIND $rows AS row
MATCH (ch:Chunk {id: row.id})
SET ch.embedding = row.embedding
"""


def reembed(db, embeddings, batchSize=BATCH_SIZE):
    chunks = db.executeQuery(queryFetchChunks)
    total = len(chunks)
    print(f"{total} chunks para reprocessar com {EMBEDDING_MODEL}.")

    for start in range(0, total, batchSize):
        batch = chunks[start:start + batchSize]
        vectors = embeddings.embed_documents([chunk["texto"] for chunk in batch])

        db.executeQuery(queryUpdateEmbeddings, parameters={
            "rows": [
                {"id": chunk["id"], "embedding": vector}
                for chunk, vector in zip(batch, vectors)
            ]
        })

        print(f"{min(start + batchSize, total)}/{total} chunks atualizados.")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regera os embeddings de todos os Chunks do grafo."
    )
    parser.add_argument(
        "--batch-size",
        dest="batchSize",
        type=int,
        default=BATCH_SIZE,
        help=f"Quantidade de chunks por lote (padrao: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    db = Neo4jConnection()
    total = reembed(db, buildEmbeddings(), batchSize=args.batchSize)
    print(f"Pronto: {total} chunks reindexados.")
