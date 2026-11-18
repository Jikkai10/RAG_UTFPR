import json
import math
import os
import re
import uuid

from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama
from typing import List, Optional, Union
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from config import RERANKER_MODEL



def envFlag(name, default):
    """Liga/desliga uma etapa pelo ambiente, para comparar configuracoes sem editar o codigo."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def envInt(name, default):
    value = os.getenv(name)
    return default if value is None or not value.strip() else int(value)


def envFloat(name, default):
    value = os.getenv(name)
    return default if value is None or not value.strip() else float(value)


USE_MULTI_QUERY = envFlag("RAG_USE_MULTI_QUERY", True)

USE_RERANK = envFlag("RAG_USE_RERANK", True)

# Chunks pedidos ao índice vetorial por documento retornado (o índice é aproximado).
FETCH_MULTIPLIER = envInt("RAG_FETCH_MULTIPLIER", 40)

FINAL_K = envInt("RAG_FINAL_K", 10)

REF_SCORE_DECAY = 0.8

MAX_ROUNDS = 3

# Piso absoluto: só descarta o lote quando nada é minimamente relevante.
MIN_RERANK_SCORE = 0.01


RERANK_SCORE_RATIO = 0.3

# Peso do cross-encoder no score final; o resto é a similaridade de embedding.
RERANK_CROSS_WEIGHT = envFloat("RAG_RERANK_CROSS_WEIGHT", 0.7)

# Quantos documentos sobrevivem ao corte final em cada busca.
ARTICLE_TOP_N = envInt("RAG_ARTICLE_TOP_N", 5)

EVENT_TOP_N = envInt("RAG_EVENT_TOP_N", 5)

EVENT_FETCH_K = 50

# Tamanho do lote que cada retriever devolve por consulta.
ARTICLE_K = envInt("RAG_ARTICLE_K", 10)

EVENT_K = envInt("RAG_EVENT_K", 10)

# Janela de contexto do ollama.
NUM_CTX = envInt("RAG_NUM_CTX", 4096)

# Divide a pergunta em subperguntas, cada uma com busca e resposta própria.
USE_DECOMPOSE = envFlag("RAG_USE_DECOMPOSE", False)

MAX_SUBQUESTIONS = envInt("RAG_MAX_SUBQUESTIONS", 4)


MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Você reescreve a pergunta de um aluno em consultas de busca para um índice vetorial de regulamentos da UTFPR.

    Regras:
    - Gere 3 consultas curtas, cada uma sobre um único assunto.
    - Escreva cada consulta como se fosse um trecho do próprio regulamento: impessoal, na terceira pessoa, no tom em que uma norma enuncia uma regra. Não use o jeito do aluno de perguntar.
    - Se a pergunta juntar vários assuntos ou vários documentos, separe: uma consulta por assunto, cada uma sozinha.
    - Não use nomes nem listas de documentos: eles dominam a similaridade e afogam o termo que interessa.
    - Não invente números, datas nem identificadores de norma que não estejam na pergunta.
    - Responda só com as consultas, uma por linha, sem numerar e sem comentar.

    Pergunta: {question}""",
)


CALENDAR_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Você reescreve a pergunta de um aluno em consultas de busca para um índice vetorial do calendário acadêmico da UTFPR.

    Regras:
    - Gere 3 consultas curtas, cada uma sobre um único evento ou prazo.
    - Escreva cada consulta como o evento é nomeado no calendário: expressão curta e impessoal, sem verbo de pergunta e sem o jeito do aluno de perguntar.
    - Se a pergunta juntar vários eventos ou prazos, separe: uma consulta por evento, cada uma sozinha.
    - Não escreva ano, período letivo nem campus: eles já são filtrados à parte e na consulta só afogam o nome do evento.
    - Não invente datas, eventos nem prazos que não estejam na pergunta.
    - Responda só com as consultas, uma por linha, sem numerar e sem comentar.

    Pergunta: {question}""",
)


DECOMPOSE_PROMPT = PromptTemplate(
    input_variables=["question", "max"],
    template="""Você divide a pergunta de um aluno em subperguntas que precisam ser respondidas para responder a pergunta inteira.

    Regras:
    - Cada subpergunta trata de uma parte da pergunta que precisa ser respondida.
    - Garanta que cada subpergunta tenha todas as informações necessárias para ser respondida sozinha.
    - Não acrescente assuntos que a pergunta não pede e não repita a mesma parte com outras palavras.
    - Faça a pergunta de forma genérica, sem se referir a documentos, artigos, normas ou dados específicos: o modelo de linguagem vai buscar os documentos e artigos que respondem a cada subpergunta.
    - Cubra todas as partes e informações necessárias da pergunta. Se ela tiver uma parte e necessitar de uma unica informação, devolva a própria pergunta.
    - Gere no máximo {max} subperguntas.
    - Responda só com as subperguntas, uma por linha, sem numerar e sem comentar.

    Exemplo:
    Pergunta: Eu reprovo com 50% numa disciplina e a segunda chamada de prova pode ser [condição]?
    Quantas faltas um aluno pode ter numa disciplina?
    A segunda chamada de prova pode ser [condição]?

    Pergunta: {question}""",
)

TERM_INPUT_PATTERNS = (
    re.compile(r"(?:^|/)\s*([1-4])\s*[ºo]?\s*$"),
    re.compile(r"([1-4])\s*[ºo]?\s*per[íi]odo", re.IGNORECASE),
)
YEAR_INPUT_RE = re.compile(r"\b(20\d{2})\b")

# Todo documento é da UTFPR: o nome da instituição na consulta só atrapalha a busca.
INSTITUTION_RE = re.compile(
    r"\s*\b(?:(?:d|n|pel)[aoe]s?\s+|para\s+a\s+)?"
    r"(?:UTFPR|Universidade\s+Tecnol[óo]gica\s+Federal\s+do\s+Paran[áa])\b",
    re.IGNORECASE,
)


def searchableQuery(query: str) -> str:
    """Consulta sem o nome da instituição; se não sobrar nada, fica a original."""

    cleaned = INSTITUTION_RE.sub("", query or "")
    cleaned = re.sub(r"\s+([?.,;:])", r"\1", re.sub(r"\s{2,}", " ", cleaned)).strip()

    return cleaned or query


def normalizeTerm(term, year):

    text = str(term or "")

    if year is None:
        match = YEAR_INPUT_RE.search(text)
        year = int(match.group(1)) if match else None
    elif not isinstance(year, int):
        match = YEAR_INPUT_RE.search(str(year))
        year = int(match.group(1)) if match else None

    number = None
    for pattern in TERM_INPUT_PATTERNS:
        match = pattern.search(text)
        if match:
            number = match.group(1)
            break

    return number, year


def formatDate(start, end):

    if not start:
        return None

    def toBrazilianDate(iso):
        year, month, day = iso.split("-")
        return f"{day}/{month}/{year}"

    return toBrazilianDate(start) if not end or end == start else f"{toBrazilianDate(start)} a {toBrazilianDate(end)}"


class Neo4jArticleRetriever(BaseRetriever):
    embeddingModel: any
    db: any
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:

        embedding = self.embeddingModel.embed_query(searchableQuery(query))

        cypher = """
        CALL db.index.vector.queryNodes(
            'content_chunk_embedding',
            $fetch_k,
            $embedding
        )
        YIELD node AS ch, score

        MATCH (ct:Content)-[:HAS_CHUNK]->(ch)

        WITH ct AS seed, max(score) AS seed_score
        ORDER BY seed_score DESC
        LIMIT $k

        OPTIONAL MATCH (seed)-[:REFERENCES]-(ref:Content)
        OPTIONAL MATCH (ref)-[:HAS_CHUNK]->(rc)

        WITH seed, seed_score, ref, max(
            CASE WHEN rc IS NULL THEN NULL
            ELSE vector.similarity.cosine(rc.embedding, $embedding) END
        ) AS ref_score

        WITH seed, seed_score, collect(
            CASE WHEN ref IS NULL THEN NULL
            ELSE {ct: ref, own: ref_score, herdado: seed_score} END
        ) AS refs

        UNWIND ([{ct: seed, own: seed_score, herdado: NULL}] + refs) AS candidate

        WITH collect(candidate) AS candidates
        UNWIND candidates AS candidate
        WITH candidates, min(candidate.own) AS lo, max(candidate.own) AS hi

        UNWIND candidates AS candidate
        WITH candidate.ct AS ct,
            CASE WHEN hi = lo THEN 1.0
                ELSE (candidate.own - lo) / (hi - lo) END AS own,
            CASE WHEN candidate.herdado IS NULL THEN NULL
                WHEN hi = lo THEN 1.0
                ELSE (candidate.herdado - lo) / (hi - lo) END AS herdado

        WITH ct, CASE
            WHEN herdado IS NULL THEN own
            WHEN own IS NULL THEN herdado * $ref_decay
            WHEN own > herdado * $ref_decay THEN own
            ELSE herdado * $ref_decay
        END AS score

        WITH ct, max(score) AS score
        ORDER BY score DESC
        LIMIT $final_k

        OPTIONAL MATCH (res:Content)-[:REF_NORM]->(ct)
        OPTIONAL MATCH (resDoc:Document {id: res.documento_id})

        WITH ct, score, collect(
            CASE WHEN res IS NULL THEN NULL
            ELSE {texto: res.texto, fonte: resDoc.titulo} END
        ) AS resolucoes

        OPTIONAL MATCH (s:Section)-[:HAS_CONT]->(ct)
        OPTIONAL MATCH (c1:Chapter)-[:HAS_CONT]->(ct)
        OPTIONAL MATCH (c2:Chapter)-[:HAS_SEC]->(s)

        WITH ct,
            coalesce(c1, c2) AS chapter,
            s,
            resolucoes,
            score

        OPTIONAL MATCH (d:Document)-[:HAS_CAP]->(chapter)

        RETURN
            score,
            ct.id AS id,
            ct.texto AS texto,
            ct.tipo AS tipo,
            ct.num AS numero,
            ct.pagina_inicio AS pagina,
            resolucoes,
            ct.alterado_por AS alterado_por,
            chapter.capitulo AS capitulo,
            s.secao AS secao,
            d.titulo AS documento
        ORDER BY score DESC;
        """

        results = self.db.executeQuery(cypher, parameters={
            "embedding": embedding,
            "k": self.k,
            "fetch_k": self.k * FETCH_MULTIPLIER,
            "final_k": FINAL_K,
            "ref_decay": REF_SCORE_DECAY
        })

        docs = []

        for record in results:
            text = record["texto"]
            # A resolução inteira só vai junto quando a alteração não pôde ser aplicada no artigo.
            applied = record["alterado_por"] or []
            for resolution in record["resolucoes"]:
                if resolution["texto"] and resolution["fonte"] not in applied:
                    text += "\n Esse artigo foi alterado: \n" + resolution["texto"]
            contentType = record["tipo"]

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": record["id"],
                        "source_type": "regulamento",
                        "Artigo": record["numero"] if contentType == "artigo" else None,
                        "Capitulo": record["capitulo"],
                        "Secao": record["secao"] if record["secao"] else None,
                        "Documento": record["documento"],
                        "Pagina": record["pagina"],
                        "Alterado por": ", ".join(record["alterado_por"]) if record["alterado_por"] else None,

                    }
                )
            )

        return docs


class Neo4jEventRetriever(BaseRetriever):
    embeddingModel: any
    db: any
    k: int = 10
    fetchK: int = EVENT_FETCH_K

    def search(
        self,
        query: str,
        term: str = None,
        campus: str = None,
        year: int = None,
        calem: bool = False,
    ) -> List[Document]:

        embedding = self.embeddingModel.embed_query(searchableQuery(query))
        term, year = normalizeTerm(term, year)

        cypher = """
        CALL db.index.vector.queryNodes(
            'event_chunk_embedding',
            $fetch_k,
            $embedding
        )
        YIELD node AS ch, score

        OPTIONAL MATCH (it:EventItem)-[:HAS_CHUNK]->(ch)
        OPTIONAL MATCH (mes:EventMonth)-[:HAS_ITEM]->(it)
        OPTIONAL MATCH (ev_secao:Events)-[:HAS_CHUNK]->(ch)
        OPTIONAL MATCH (ev_mes:Events)-[:HAS_MONTH]->(mes)

        WITH ch, score, mes, coalesce(ev_secao, ev_mes) AS ev
        WHERE ev IS NOT NULL
          AND ($periodo IS NULL OR coalesce(mes.periodo, ev.periodo) = $periodo)
          AND ($ano IS NULL OR coalesce(mes.ano, ev.ano) = $ano)
          AND ($campus IS NULL OR toLower(coalesce(ev.campus, '')) CONTAINS toLower($campus))
          // O CALEM tem calendario proprio com os mesmos nomes de evento da
          // graduacao ("Inicio do periodo letivo" em 12/08 contra 18/08), e a
          // busca devolvia as duas datas misturadas. Ou e so CALEM, ou e tudo
          // menos o CALEM.
          AND ((toUpper(coalesce(ev.categoria, '')) = 'CALEM') = $calem)

        // Ordena antes de agregar para o head(collect(...)) pegar o texto do chunk
        // de maior score, e nao um qualquer do grupo.
        WITH coalesce(mes, ev) AS alvo, mes, ev, ch, score
        ORDER BY score DESC

        WITH alvo, mes, ev, max(score) AS score, head(collect(ch.texto)) AS chunk_texto
        ORDER BY score DESC
        LIMIT $k

        OPTIONAL MATCH (alvo)-[:HAS_ITEM]->(item:EventItem)

        WITH alvo, mes, ev, score, chunk_texto,
            min(item.data_inicio) AS data_inicio,
            max(item.data_fim) AS data_fim

        OPTIONAL MATCH (d:Document)-[:HAS_EVENT]->(ev)

        WITH alvo, mes, ev, score, chunk_texto, data_inicio, data_fim,
            head(collect(d.titulo)) AS documento

        RETURN
            score,
            alvo.id AS id,
            // Quando o alvo e um mes, o texto dele e o conteudo util. Quando e o
            // Events (periodo inteiro), o chunk que casou e o resumo dos dias
            // letivos numa linha, enquanto alvo.texto espalha os mesmos numeros
            // por dezenas de linhas de datas: o modelo perde meses no caminho.
            CASE WHEN mes IS NULL THEN chunk_texto ELSE alvo.texto END AS texto,
            toString(data_inicio) AS data_inicio,
            toString(data_fim) AS data_fim,
            mes.mes AS mes,
            mes.dias_letivos AS dias_letivos,
            coalesce(mes.periodo, ev.periodo) AS periodo,
            coalesce(mes.ano, ev.ano) AS ano,
            ev.categoria AS categoria,
            ev.campus AS campus,
            documento
        ORDER BY score DESC;
        """

        results = self.db.executeQuery(cypher, parameters={
            "embedding": embedding,
            "k": self.k,
            "fetch_k": self.fetchK,
            "periodo": term,
            "campus": campus or None,
            "ano": year,
            "calem": bool(calem),
        })

        docs = []

        for record in results:
            if not record["texto"]:
                print(f"Evento {record['id']} ignorado: sem texto no grafo.")
                continue

            docs.append(
                Document(
                    page_content=record["texto"],
                    metadata={
                        "id": record["id"],
                        "source_type": "calendario",
                        "Mes": record["mes"],
                        "Dias letivos": record["dias_letivos"],
                        "Categoria": record["categoria"],
                        "Periodo": f"{record['periodo']}º" if record["periodo"] else None,
                        "Ano": record["ano"],
                        "Campus": record["campus"],
                        "Documento": record["documento"],

                    }
                )
            )

        return docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.search(query)


class RAGSources(BaseModel):
    sources: List[int] = Field(
        description="Lista dos IDs dos documentos necessários"
    )

class MyState(MessagesState):
    rounds: int
    candidates: List[dict]
    question: str

class Rag:

    def __init__(self, db, embeddingModel, llm):

        self.db = db
        self.embeddingModel = embeddingModel
        self.llm = llm

        # Registra a configuração usada em cada rodada da avaliação.
        print(
            "config do RAG: "
            f"multi_query={USE_MULTI_QUERY} rerank={USE_RERANK} "
            f"article_k={ARTICLE_K} final_k={FINAL_K} article_top_n={ARTICLE_TOP_N} "
            f"event_k={EVENT_K} event_top_n={EVENT_TOP_N} num_ctx={NUM_CTX} "
            f"decompose={USE_DECOMPOSE} max_subquestions={MAX_SUBQUESTIONS} "
            f"cross_weight={RERANK_CROSS_WEIGHT}"
        )

        # temperature 0 e seed fixo deixam a escolha da busca reprodutível.
        self.searchLlm = ChatOllama(
            model=getattr(llm, "model", "llama3.1"),
            base_url=getattr(llm, "base_url", None),
            temperature=0,
            seed=0,
            num_ctx=NUM_CTX,
        )



        self.articleRetriever = Neo4jArticleRetriever(
            embeddingModel=self.embeddingModel, db=self.db, k=ARTICLE_K
        )
        self.eventRetriever = Neo4jEventRetriever(
            embeddingModel=self.embeddingModel, db=self.db, k=EVENT_K
        )
        # O calendário tem multi-query próprio (searchCalendarDocs) para manter os filtros.
        self.articleMultiQuery = MultiQueryRetriever.from_llm(
            retriever=self.articleRetriever,
            llm=self.searchLlm,
            prompt=MULTI_QUERY_PROMPT,
            include_original=True,
        )
        self.calendarQueryChain = (
            CALENDAR_MULTI_QUERY_PROMPT | self.searchLlm | StrOutputParser()
        )
        self.rerankerModel = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL
        )

        self.tools = [
            tool(response_format="content_and_artifact")(self.searchRegulations),
            tool(response_format="content_and_artifact")(self.searchCalendar),
        ]

        # Cada subpergunta passa por um agente que busca e escreve a resposta parcial.
        subBuilder = StateGraph(MyState)
        subBuilder.add_node("searchOrAnswer", self.searchOrAnswer)
        subBuilder.add_node("tools", self.numberedTools)
        subBuilder.add_node("forceAnswer", self.forceAnswer)
        subBuilder.set_entry_point("searchOrAnswer")
        subBuilder.add_conditional_edges(
            "searchOrAnswer",
            self.shouldContinue,
            {"tools": "tools", "answer": "forceAnswer"},
        )
        subBuilder.add_edge("tools", "searchOrAnswer")
        subBuilder.add_edge("forceAnswer", END)
        self.subquestionGraph = subBuilder.compile()

        self.decomposeChain = DECOMPOSE_PROMPT | self.searchLlm | StrOutputParser()

    def querySimilarity(self, query: str, docs: List[Document]) -> List[float]:
        """Similaridade da consulta com cada documento, normalizada no lote."""
        print(f"Query: {query}", flush=True)
        queryVector = self.embeddingModel.embed_query(query)
        docVectors = self.embeddingModel.embed_documents(
            [d.page_content for d in docs]
        )

        queryNorm = math.sqrt(sum(v * v for v in queryVector)) or 1.0

        similarities = []
        for vector in docVectors:
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            dot = sum(a * b for a, b in zip(queryVector, vector))
            similarities.append(dot / (queryNorm * norm))

        lo, hi = min(similarities), max(similarities)

        if hi == lo:
            return [1.0] * len(similarities)

        return [(s - lo) / (hi - lo) for s in similarities]

    def rerank(
        self,
        query: str,
        docs: List[Document],
        topN: int = ARTICLE_TOP_N,
        minScore: float = MIN_RERANK_SCORE,
        ratio: float = RERANK_SCORE_RATIO,
    ) -> List[Document]:

        if not docs:
            return []

        # Sem reranker, a ordem da busca vetorial já é por score.
        if not USE_RERANK:
            return docs[:topN]

        query = searchableQuery(query)
        scores = self.rerankerModel.score([
            (query, d.page_content) for d in docs
        ])

        if RERANK_CROSS_WEIGHT < 1.0:
            scores = [
                RERANK_CROSS_WEIGHT * cross + (1 - RERANK_CROSS_WEIGHT) * sim
                for cross, sim in zip(scores, self.querySimilarity(query, docs))
            ]

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        floor = max(minScore, ranked[0][1] * ratio)

        return [doc for doc, score in ranked[:topN] if score >= floor]

    def generateQueries(self, chain, query: str) -> List[str]:
        """Pergunta original mais as reescritas do LLM, sem repetidas."""

        queries = [query]
        seen = {query.strip().lower()}

        try:
            text = chain.invoke({"question": query})
        except Exception as error:
            # Se a reescrita falhar, segue só com a pergunta original.
            print(f"multi_query falhou, seguindo so com a pergunta original: {error}")
            return queries

        for line in text.splitlines():
            # Mesmo mandado nao numerar, o modelo as vezes marca a lista.
            line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            key = line.lower()

            if not line or key in seen:
                continue

            seen.add(key)
            queries.append(line)

        return queries

    def searchCalendarDocs(
        self,
        query: str,
        term: str = None,
        campus: str = None,
        year: int = None,
        calem: bool = False,
    ) -> List[Document]:
        """Busca no calendario com as variantes da pergunta, mantendo os filtros."""

        if not USE_MULTI_QUERY:
            return self.eventRetriever.search(
                query, term=term, campus=campus, year=year, calem=calem
            )

        docs = []
        seen = set()

        for variant in self.generateQueries(self.calendarQueryChain, query):
            print(f"  variante do calendário: {variant}")

            for doc in self.eventRetriever.search(
                variant, term=term, campus=campus, year=year, calem=calem
            ):
                key = doc.metadata.get("id") or doc.page_content
                if key in seen:
                    continue

                seen.add(key)
                docs.append(doc)

        return docs

    def docLabel(self, doc: Document) -> str:
        """Rótulo legível do documento, que o modelo vê antes do conteúdo."""
        meta = doc.metadata

        if meta.get("source_type") == "calendario":
            period = f"{meta['Periodo']} período" if meta.get("Periodo") else None
            if period and meta.get("Ano"):
                period += f" de {meta['Ano']}"
            details = [
                meta.get("Campus"),
                period,
                meta.get("Mes"),
                meta.get("Categoria"),
                f"dias letivos: {meta['Dias letivos']}" if meta.get("Dias letivos") else None,
            ]
            label = meta.get("Documento") or "Calendário acadêmico"
            details = [part for part in details if part]
            return f"{label} ({', '.join(details)})" if details else label

        where = f"Art. {meta['Artigo']}" if meta.get("Artigo") else "Trecho"
        label = f"{where} do {meta.get('Documento') or 'regulamento'}"

        details = []
        if meta.get("Capitulo") and meta["Capitulo"] != "—":
            details.append(f"Capítulo {meta['Capitulo']}")
        if meta.get("Secao") and meta["Secao"] != "—":
            details.append(f"Seção {meta['Secao']}")
        if meta.get("Alterado por"):
            details.append(f"redação alterada por {meta['Alterado por']}")

        return f"{label} ({', '.join(details)})" if details else label

    def serialize(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"{self.docLabel(doc)}\n{doc.page_content}"
            for doc in docs
        )

    def collectDocs(self, messages) -> List[Document]:

        docs = []
        seen = set()

        for message in messages:
            if message.type != "tool":
                continue

            for doc in getattr(message, "artifact", None) or []:
                key = doc.metadata.get("id") or doc.page_content
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)

        return docs

    def searchRegulations(self, query: str):
        """Busca trechos dos regulamentos e normas oficiais da UTFPR: regras, direitos, deveres, critérios, penalidades e prazos.
        Gere uma entrada "query" com uma frase completa que traga todos os detalhes do que se procura (assunto, situação, condição) em texto corrido como uma pergunta."""
        print(f"Retrieving regulations for query: {query}")
        retriever = self.articleMultiQuery if USE_MULTI_QUERY else self.articleRetriever
        retrievedDocs = self.rerank(query, retriever.invoke(query))
        print(f"Retrieved {len(retrievedDocs)} regulation documents.")
        if not retrievedDocs:
            return "Nenhum resultado encontrado.", []

        return self.serialize(retrievedDocs), retrievedDocs

    def searchCalendar(
        self,
        query: str,
        # O modelo manda o período ora como "1", ora como 1; normalizeTerm aceita os dois.
        term: Optional[Union[str, int]] = None,
        campus: Optional[str] = None,
        year: Optional[int] = None,
        calem: bool = False,
    ):
        """Busca datas e eventos do calendário acadêmico da UTFPR (matrícula, início e fim das aulas, provas, férias, feriados, prazos de solicitação).
        Use sempre que a pergunta envolver "quando", datas, prazos ou períodos letivos.
        Gere uma entrada "query": "frase para busca RAG", apenas a string.
        Opcionalmente informe "term" com o número do período letivo ("1" ou "2"), "year" com o ano do calendário (ex: 2026) e "campus" com o nome do campus (ex: "Toledo").
        "calem" escolhe o calendário do CALEM, que tem datas próprias (ex: teste de nivelamento do CALEM): false (padrão) busca no calendário da graduação, true só no do CALEM. Use true apenas se a pergunta citar o CALEM.
        Só filtre pelo que a pergunta disser: filtro errado devolve zero resultados."""
        print(f"Retrieving calendar events for query: {query} (term={term}, campus={campus}, year={year}, calem={calem})")
        retrievedDocs = self.rerank(
            query,
            self.searchCalendarDocs(query, term=term, campus=campus, year=year, calem=calem),
            topN=EVENT_TOP_N,
        )
        print(f"Retrieved {len(retrievedDocs)} calendar documents.")
        if not retrievedDocs:
            return "Nenhum resultado encontrado.", []

        return self.serialize(retrievedDocs), retrievedDocs


    def shouldContinue(self, state: MyState):

        lastMessage = state["messages"][-1]

        if not getattr(lastMessage, "tool_calls", None):
            return "answer"

        if state.get("rounds", 0) > MAX_ROUNDS:
            print(f"Limite de {MAX_ROUNDS} rodadas atingido, gerando a resposta.")
            return "answer"

        return "tools"

    def getRecentMessages(self, threadId, limit=10):

        query = """
        MATCH (c:Chat {thread_id:$thread_id})-[:HAS_MESSAGE]->(m)

        RETURN m.role as role, m.content as content
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """

        result = self.db.executeQuery(
            query, parameters = {
                "thread_id": threadId,
                "limit": limit
            }
        )

        history = [
                {"role": r["role"], "content": r["content"]}
                for r in result
            ][::-1]


        return history

    def saveMessage(self, threadId, role, content, sources = [], contexts = []):

        query = """
        MERGE (c:Chat {thread_id:$thread_id})

        CREATE (m:Message {
            id: randomUUID(),
            role: $role,
            content: $content,
            sources: $sources,
            contexts: $contexts,
            timestamp: datetime()
        })

        MERGE (c)-[:HAS_MESSAGE]->(m)
        """

        self.db.executeQuery(
            query, parameters = {
                "thread_id": threadId,
                "role": role,
                "content": content,
                "sources": sources,
                "contexts": contexts
            }
        )


    async def updateChat(self, threadId: str, message: str):

        prompt = f"""
        Crie um título de chat curto e direto (maximo de 5 palavras) para esta pergunta.
        Escreva apenas o titulo, sem "Aqui está", "O titulo é" ou similares.

        Pergunta:
        {message}

        Título:
        """

        response = self.llm.invoke(prompt)
        title = response.content
        query="""
        MATCH (c:Chat {thread_id: $thread_id})
        SET c.title = $new_title
        RETURN c
        """

        self.db.executeQuery(
            query, parameters = {
                "thread_id": threadId,
                "new_title": title
            }
        )

        return title


    async def usedSources(self, answer: str, candidates: List[dict]):

        if not candidates:
            return []

        # Aceita citações como [3], [1, 3] e [7, Parágrafo único]; só os números contam.
        cited = []
        for group in re.findall(r"\[([^\[\]]+)\]", answer or ""):
            for number in (part.strip() for part in group.split(",")):
                if not number.isdigit():
                    continue
                index = int(number) - 1
                if 0 <= index < len(candidates) and index not in cited:
                    cited.append(index)

        if cited:
            return [candidates[index] for index in cited]

        return await self.askUsedSources(answer, candidates)

    async def askUsedSources(self, answer: str, candidates: List[dict]):

        numbered = "\n\n".join(
            f"[{i}] {candidate['metadata']}\n{candidate['content'][:500]}"
            for i, candidate in enumerate(candidates, start=1)
        )

        prompt = f"""
        Abaixo estão documentos numerados e uma resposta gerada a partir deles.
        Retorne os números dos documentos que foram efetivamente usados na resposta.

        Documentos:
        {numbered}

        Resposta:
        {answer}
        """

        try:
            result = await self.llm.with_structured_output(RAGSources).ainvoke(prompt)
            selected = [
                candidates[number - 1]
                for number in result.sources
                if 0 < number <= len(candidates)
            ]
            if selected:
                return selected
        except Exception as error:
            print(f"Não foi possível identificar as fontes usadas: {error}")

        return candidates

    def docKey(self, doc: Document):
        return doc.metadata.get("id") or doc.page_content

    def decompose(self, question: str) -> List[str]:
        """Subperguntas que, respondidas, cobrem a pergunta inteira."""

        try:
            text = self.decomposeChain.invoke(
                {"question": question, "max": MAX_SUBQUESTIONS}
            )
        except Exception as error:
            print(f"decomposicao falhou, seguindo so com a pergunta original: {error}")
            return [question]

        lines = []
        seen = set()

        for line in text.splitlines():
            line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            key = line.lower()

            if not line or key in seen:
                continue

            seen.add(key)
            lines.append(line)

        # Descarta linhas que não são perguntas (ex.: "Subperguntas:").
        questions = [line for line in lines if line.endswith("?")] or lines

        return questions[:MAX_SUBQUESTIONS] or [question]

    def candidateKey(self, candidate: dict):
        return candidate["metadata"].get("id") or candidate["content"]

    def subquestionSystemPrompt(self, question: str, subquestion: str = None) -> str:

        # Numa subpergunta, a pergunta completa vai junto como contexto.
        intro = "Você responde a pergunta do usuário sobre os documentos oficiais."
        search = "Caso a pergunta tenha mais de um assunto, faça uma busca para cada assunto."
        if subquestion and subquestion.strip() != question.strip():
            intro += f"""

        A pergunta que você recebeu é parte de uma pergunta maior do usuário, abaixo. Use a pergunta maior para completar o que a sua pergunta deixar implícito, como condições e tipo de informação, nas buscas e na resposta, mas responda só a pergunta que você recebeu.
        Pergunta completa do usuário: {question}"""
        
            search = "Verifique na pergunta completa se há condições, exceções ou detalhes que a subpergunta não menciona, e use-os na busca e na resposta."

        date = datetime.now().strftime("%d/%m/%Y")
        return f"""
        Data atual: {date}\n\n
        {intro}

        Você tem duas ferramentas de busca:
        - searchRegulations: regras, direitos, deveres, critérios, prazos gerais que não estão no calendário (busque primeiro no calendário).
        - searchCalendar: datas, prazos e eventos. Busca no calendário da graduação; só passe calem=true se a pergunta for sobre o CALEM.

        Busca:
        - Sempre use as ferramentas antes de responder e use quantas forem necessárias.
        - Se a resposta da subpergunta não vier, faça uma nova busca usando outros termos.
        - Se a subpergunta envolver datas e eventos, mas não especificar o período letivo, faça uma busca por período (1 e 2).
        - Se a subpergunta se tratar de datas, mas o calendário não deixar explicita, busque no regulamento.
        - {search}

        Resposta:
        - Responda só o que a subpergunta pede, de forma direta.
        - No que a subpergunta pede, seja completo e preciso: reproduza números, prazos, percentuais, condições, exceções e incisos exatamente como estão nos documentos.
        - Os documentos devolvidos pelas buscas vêm como "Documento [3]: Art. 55 do ...". Cite o número do documento entre colchetes logo depois da informação, por exemplo: "o prazo é de 5 (cinco) dias úteis [3]". O número entre colchetes é sempre o do documento, nunca o do artigo: para o Documento [3], que é o Art. 55, escreva [3], e não [55] nem [Art. 55]. Nunca escreva "[n]".
        - Responda usando apenas o que está nos documentos e diga explicitamente o que a subpergunta pede e não consta neles.
        - Caso a subpergunta envolva datas, mas não especifique se é o primeiro ou segundo semestre, responda com a data de ambos os semestres.
        - Nunca invente números e não escreva introdução nem conclusão.
        """

    def searchOrAnswer(self, state: MyState):
        llmWithTools = self.searchLlm.bind_tools(self.tools)
        subquestion = state["messages"][0].content
        prompt = [
            SystemMessage(content=self.subquestionSystemPrompt(state.get("question") or subquestion, subquestion))
        ] + state["messages"]

        response = llmWithTools.invoke(prompt)

        # Força ao menos uma busca por subpergunta.
        searched = any(message.type == "tool" for message in state["messages"])

        if not searched and not response.tool_calls:
            print("  agente respondeu sem buscar, pedindo uma busca")
            response = llmWithTools.invoke(prompt + [
                response,
                HumanMessage("Você ainda não fez nenhuma busca. Antes de responder, chame uma das ferramentas de busca."),
            ])

        if not searched and not response.tool_calls:
            print("  agente seguiu sem buscar, buscando pela subpergunta")
            response = AIMessage(content="", tool_calls=[{
                "name": "searchRegulations",
                "args": {"query": subquestion},
                "id": str(uuid.uuid4()),
                "type": "tool_call",
            }])

        usage = getattr(response, "usage_metadata", None) or {}
        print(f"  agente da subpergunta: {usage.get('input_tokens')} tokens de entrada, "
              f"tool_calls={response.tool_calls}")
        return {"messages": [response], "rounds": state.get("rounds", 0) + 1}

    def numberedTools(self, state: MyState):
        """Roda as buscas pedidas e numera os documentos com ids únicos na pergunta inteira."""
        candidates = list(state.get("candidates") or [])
        ids = {
            self.candidateKey(candidate): position
            for position, candidate in enumerate(candidates, start=1)
        }
        toolsByName = {t.name: t for t in self.tools}

        messages = []

        for call in state["messages"][-1].tool_calls:
            selected = toolsByName.get(call["name"])

            if selected is None:
                messages.append(ToolMessage(
                    content=f"Ferramenta {call['name']} não existe.",
                    tool_call_id=call["id"],
                    name=call["name"],
                ))
                continue

            # searchCalendar sem query usa a própria subpergunta.
            args = call.get("args") or {}
            if not str(args.get("query") or "").strip():
                print(f"  busca {call['name']} sem query, usando a subpergunta")
                call = {**call, "args": {**args, "query": state["messages"][0].content}}

            # Argumento inválido volta como erro para o agente.
            try:
                toolMessage = selected.invoke({**call, "type": "tool_call"})
            except Exception as error:
                print(f"  busca {call['name']} falhou: {error}")
                messages.append(ToolMessage(
                    content=f"A busca falhou: {error}. Corrija os argumentos e tente de novo.",
                    tool_call_id=call["id"],
                    name=call["name"],
                ))
                continue

            docs = getattr(toolMessage, "artifact", None) or []

            for doc in docs:
                key = self.docKey(doc)
                if key not in ids:
                    candidates.append({"content": doc.page_content, "metadata": doc.metadata})
                    ids[key] = len(candidates)

            if docs:
                toolMessage.content = "\n\n".join(
                    f"Documento [{ids[self.docKey(doc)]}]: {self.docLabel(doc)}\n{doc.page_content}"
                    for doc in docs
                )

            messages.append(toolMessage)

        return {"messages": messages, "candidates": candidates}

    def forceAnswer(self, state: MyState):
        """Garante a resposta parcial quando o limite de rodadas corta o agente no meio de uma busca."""

        lastMessage = state["messages"][-1]

        if not getattr(lastMessage, "tool_calls", None):
            return {}

        question = state.get("question") or state["messages"][0].content

        response = self.searchLlm.invoke(
            [SystemMessage(content=self.subquestionSystemPrompt(question, state["messages"][0].content))]
            + state["messages"][:-1]
            + [HumanMessage("Responda agora a subpergunta com o que as buscas já encontraram.")]
        )
        return {"messages": [response]}

    def finalPrompt(self, question: str, partials: List[tuple]) -> list:

        partialsContent = "\n\n".join(
            f"### Subpergunta {i}: {subquestion}\n{partial}"
            for i, (subquestion, partial) in enumerate(partials, start=1)
        )

        date = datetime.now().strftime("%d/%m/%Y")
        systemMessageContent = f"""
        Data atual: {date}\n\n
        Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da UTFPR.
        A pergunta do usuário foi dividida em subperguntas, e cada uma já foi respondida a partir dos documentos oficiais. As subperguntas e as respostas parciais abaixo são só o contexto para você responder: considere-as verdade, mas não responda as subperguntas uma a uma.
        - Responda apenas o que a pergunta do usuário pede, usando as respostas parciais como fonte, sem repetir a mesma informação.
        - Não perca nenhum item das respostas parciais que a pergunta pede: se uma resposta parcial lista casos, condições, documentos ou prazos, todos eles entram na resposta final, com os números, percentuais e exceções.
        - Se uma resposta parcial disser que algo não consta e outra trouxer essa informação, use a informação.
        - Seja direto: deixe de fora o que a pergunta não pede, mesmo que esteja nas respostas parciais, e não repita a mesma informação em mais de um lugar.
        - Mantenha os números de citação de cada informação ([1], [3]...) exatamente como aparecem nas respostas parciais, sem renumerar e sem criar números novos. Nunca use número de artigo como citação e nunca escreva "[n]".
        - Responda usando apenas o que está nas respostas parciais. Se uma parte da pergunta não consta nelas, diga explicitamente qual parte falta, sem descartar o resto da resposta.
        - Nunca invente números.
        - Escreva em markdown direto no texto da resposta, sem colocar a resposta dentro de bloco de código (```).


        Contexto (subperguntas e respostas parciais):
        \n\n
        {partialsContent}
        """
        print(f"prompt final:\n{systemMessageContent}", flush=True)
        return [SystemMessage(systemMessageContent), HumanMessage(question)]

    async def runDecomposed(self, message, sessionId, persist=True):

        if persist:
            self.saveMessage(sessionId, "user", message)

        # Sem decomposição, a própria pergunta é a única subpergunta.
        subquestions = self.decompose(message) if USE_DECOMPOSE else [message]
        print(f"subperguntas: {subquestions}")

        # Lista única de documentos: a posição (1-based) é o id citado.
        candidates = []
        partials = []

        for subquestion in subquestions:
            print(f"subpergunta: {subquestion}")
            result = await self.subquestionGraph.ainvoke({
                "messages": [{"role": "user", "content": subquestion}],
                "rounds": 0,
                "candidates": candidates,
                "question": message,
            })

            for toolMessage in result["messages"]:
                if toolMessage.type == "tool":
                    yield ("searching", {
                        "tool": toolMessage.name,
                        "encontrados": len(getattr(toolMessage, "artifact", None) or []),
                    })

            candidates = result.get("candidates") or candidates
            partials.append((subquestion, result["messages"][-1].content))

        yield ("steps", [
            {"subquestion": subquestion, "answer": partial}
            for subquestion, partial in partials
        ])

        fullAnswer = ""

        # Com uma subpergunta só, a resposta parcial já é a final.
        if len(partials) == 1:
            fullAnswer = partials[0][1]
            yield ("token", fullAnswer)
        else:
            # A junção usa o modelo determinístico das buscas.
            async for chunk in self.searchLlm.astream(self.finalPrompt(message, partials)):
                content = getattr(chunk, "content", None)

                if content:
                    fullAnswer += content
                    yield ("token", content)

        sources = await self.usedSources(fullAnswer, candidates)

        if persist:
            self.saveMessage(
                sessionId,
                "assistant",
                fullAnswer,
                sources=json.dumps(sources),
                contexts=json.dumps(candidates)
            )

        yield ("contexts", candidates)

        yield ("sources", sources)

    async def run(self, message, sessionId, persist=True):

        async for item in self.runDecomposed(message, sessionId, persist=persist):
            yield item

    async def answer(self, message, chatHistory, sessionId, persist=True):

        fullAnswer = ""
        sources = []
        contexts = []
        steps = []

        async for kind, payload in self.run(message, sessionId, persist=persist):
            if kind == "token":
                fullAnswer += payload
            elif kind == "contexts":
                contexts = payload
            elif kind == "sources":
                sources = payload
            elif kind == "steps":
                steps = payload

        return {
            "answer": fullAnswer,
            "sources": sources,
            "contexts": contexts,
            "steps": steps
        }

    async def answerStream(self, message, chatHistory, sessionId):

        async for kind, payload in self.run(message, sessionId):

            if kind == "token":
                yield (
                    f"event: token\n"
                    f"data: {json.dumps(payload)}\n\n"
                )

            elif kind == "searching":
                yield (
                    f"event: searching\n"
                    f"data: {json.dumps(payload)}\n\n"
                )

            elif kind == "contexts":
                yield (
                    f"event: contexts\n"
                    f"data: {json.dumps(payload)}\n\n"
                )

            elif kind == "sources":
                yield (
                    f"event: sources\n"
                    f"data: {json.dumps(payload)}\n\n"
                )

        yield "event: end\ndata: done\n\n"

        if len(chatHistory) == 0:
            title = await self.updateChat(sessionId, message)

            yield (
                f"event: title\n"
                f"data: {json.dumps(title)}\n\n"
            )
