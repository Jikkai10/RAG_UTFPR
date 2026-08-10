import asyncio
import json
import re

#from langchain.schema import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
#from langgraph.checkpoint.memory import MemorySaver
#from langchain_chroma import Chroma
#from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional
#import numpy as np
from pydantic import BaseModel, Field
import logging


FETCH_MULTIPLIER = 5

FINAL_K = 20

REF_SCORE_DECAY = 0.8

MAX_ROUNDS = 3

MIN_RERANK_SCORE = 0.05

TERM_INPUT_PATTERNS = (
    re.compile(r"(?:^|/)\s*([1-4])\s*[ºo]?\s*$"),
    re.compile(r"([1-4])\s*[ºo]?\s*per[íi]odo", re.IGNORECASE),
)
YEAR_INPUT_RE = re.compile(r"\b(20\d{2})\b")


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

        embedding = self.embeddingModel.embed_query(query)

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
        OPTIONAL MATCH (s:Section)-[:HAS_CONT]->(ct)
        OPTIONAL MATCH (c1:Chapter)-[:HAS_CONT]->(ct)
        OPTIONAL MATCH (c2:Chapter)-[:HAS_SEC]->(s)

        WITH ct,
            coalesce(c1, c2) AS chapter,
            s,
            res,
            score

        OPTIONAL MATCH (d:Document)-[:HAS_CAP]->(chapter)

        RETURN
            score,
            ct.id AS id,
            ct.texto AS texto,
            ct.tipo AS tipo,
            ct.num AS numero,
            ct.pagina_inicio AS pagina,
            res.texto AS res_texto,
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
            if record["res_texto"]:
                text += "\n Esse artigo foi alterado: \n" + record["res_texto"]
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

                    }
                )
            )

        return docs


class Neo4jEventRetriever(BaseRetriever):
    embeddingModel: any
    db: any
    k: int = 5

    def search(
        self,
        query: str,
        term: str = None,
        campus: str = None,
        year: int = None,
    ) -> List[Document]:

        embedding = self.embeddingModel.embed_query(query)
        term, year = normalizeTerm(term, year)

        cypher = """
        CALL db.index.vector.queryNodes(
            'event_chunk_embedding',
            $fetch_k,
            $embedding
        )
        YIELD node AS ch, score

        OPTIONAL MATCH (it:EventItem)-[:HAS_CHUNK]->(ch)
        OPTIONAL MATCH (ev_secao:Events)-[:HAS_CHUNK]->(ch)
        OPTIONAL MATCH (ev_item:Events)-[:HAS_ITEM]->(it)

        WITH ch, score, it, coalesce(ev_secao, ev_item) AS ev
        WHERE ev IS NOT NULL
          AND ($periodo IS NULL OR coalesce(it.periodo, ev.periodo) = $periodo)
          AND ($ano IS NULL OR coalesce(it.ano, ev.ano) = $ano)
          AND ($campus IS NULL OR toLower(coalesce(ev.campus, '')) CONTAINS toLower($campus))

        WITH coalesce(it, ev) AS alvo, it, ev, ch, score
        ORDER BY score DESC

        WITH alvo, it, ev, max(score) AS score, head(collect(ch.texto)) AS chunk_texto
        ORDER BY score DESC
        LIMIT $k

        OPTIONAL MATCH (d:Document)-[:HAS_EVENT]->(ev)

        WITH alvo, it, ev, score, chunk_texto, head(collect(d.titulo)) AS documento

        RETURN
            score,
            alvo.id AS id,
            coalesce(chunk_texto, alvo.texto) AS texto,
            it.texto AS item,
            toString(it.data_inicio) AS data_inicio,
            toString(it.data_fim) AS data_fim,
            it.mes AS mes,
            coalesce(it.periodo, ev.periodo) AS periodo,
            coalesce(it.ano, ev.ano) AS ano,
            ev.categoria AS categoria,
            ev.campus AS campus,
            documento
        ORDER BY score DESC;
        """

        results = self.db.executeQuery(cypher, parameters={
            "embedding": embedding,
            "k": self.k,
            "fetch_k": self.k * FETCH_MULTIPLIER,
            "periodo": term,
            "campus": campus or None,
            "ano": year,
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
                        "Data": formatDate(record["data_inicio"], record["data_fim"]),
                        "Mes": record["mes"],
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

# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
class Rag:

    def __init__(self, db, embeddingModel, llm):

        self.db = db
        self.embeddingModel = embeddingModel
        self.llm = llm


        # class LineListOutputParser(BaseOutputParser[List[str]]):
        #     """Output parser for a list of lines."""

        #     def parse(self, text: str) -> List[str]:
        #         lines = text.strip().split("\n")
        #         return list(filter(None, lines))  # Remove empty lines


        # output_parser = LineListOutputParser()

        # QUERY_PROMPT = PromptTemplate(
        #     input_variables=["question"],
        #     template="""Você é um assistente baseado em um modelo de linguagem de IA.
        #     Sua tarefa é gerar três versões diferentes da pergunta feita pelo usuário para recuperar documentos relevantes de um banco de dados vetorial.
        #     Ao gerar múltiplas perspectivas da pergunta original, seu objetivo é ajudar o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
        #     Forneça essas perguntas alternativas separadas por quebras de linha. Retorne apenas as perguntas, sem explicações adicionais ou coisas como 'aqui estão as perguntas'.
        #     Pergunta original: {question}""",
        # )


        # llm_retriever = ChatOllama(model="llama3.1", temperature=0)
        # llm_chain = QUERY_PROMPT | llm_retriever | output_parser
        # retriever = MultiQueryRetriever(
        #     retriever=vector_store.as_retriever(), llm_chain=llm_chain, parser_key="lines", include_original=True
        # )

        self.articleRetriever = Neo4jArticleRetriever(
            embeddingModel=self.embeddingModel, db=self.db, k=10
        )
        self.eventRetriever = Neo4jEventRetriever(
            embeddingModel=self.embeddingModel, db=self.db, k=10
        )
        self.rerankerModel = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base"
        )

        self.tools = [
            tool(response_format="content_and_artifact")(self.searchRegulations),
            tool(response_format="content_and_artifact")(self.searchCalendar),
        ]
        tools = ToolNode(self.tools)

        graphBuilder = StateGraph(MyState)
        graphBuilder.add_node("queryOrRespond", self.queryOrRespond)
        graphBuilder.add_node("tools", tools)
        graphBuilder.add_node("generate", self.generate)


        graphBuilder.set_entry_point("queryOrRespond")
        graphBuilder.add_conditional_edges(
            "queryOrRespond",
            self.shouldContinue,
            {"tools": "tools", "generate": "generate"},
        )
        graphBuilder.add_edge(
            "tools",
            "queryOrRespond",
        )
        graphBuilder.add_edge(
            "generate",
            END,
        )


        self.graph = graphBuilder.compile()

        # memory = MemorySaver()
        # self.graph = graphBuilder.compile(checkpointer=memory)


    def rerank(
        self,
        query: str,
        docs: List[Document],
        topN: int = 5,
        minScore: float = MIN_RERANK_SCORE,
    ) -> List[Document]:

        if not docs:
            return []

        scores = self.rerankerModel.score([
            (query, d.page_content) for d in docs
        ])

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked[:topN] if score >= minScore]

    def docLabel(self, doc: Document) -> str:

        return ", ".join(
            f"{k}: {v}"
            for k, v in doc.metadata.items()
            if k != "id" and v is not None
        )

    def serialize(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"Fonte: {self.docLabel(doc)}\nConteudo: {doc.page_content}"
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

    #@tool(response_format="content_and_artifact")
    def searchRegulations(self, query: str):
        """Busca trechos dos regulamentos, regimentos e normas oficiais da UTFPR (artigos, capítulos e seções).
        Use para regras, direitos, deveres, critérios, penalidades e prazos regulamentares.
        Gere uma entrada "query": "frase para busca RAG", apenas a string, sem type ou outras coisas"""
        print(f"Retrieving regulations for query: {query}")
        retrievedDocs = self.rerank(query, self.articleRetriever.invoke(query))
        print(f"Retrieved {len(retrievedDocs)} regulation documents.")
        if not retrievedDocs:
            return "Nenhum resultado encontrado.", []

        return self.serialize(retrievedDocs), retrievedDocs

    #@tool(response_format="content_and_artifact")
    def searchCalendar(
        self,
        query: str,
        term: Optional[str] = None,
        campus: Optional[str] = None,
        year: Optional[int] = None,
    ):
        """Busca datas e eventos do calendário acadêmico da UTFPR (matrícula, início e fim das aulas, provas, férias, feriados, prazos de solicitação).
        Use sempre que a pergunta envolver "quando", datas, prazos ou períodos letivos.
        Gere uma entrada "query": "frase para busca RAG", apenas a string.
        Opcionalmente informe "term" com o número do período letivo ("1" ou "2"), "year" com o ano do calendário (ex: 2026) e "campus" com o nome do campus (ex: "Toledo").
        Só filtre pelo que a pergunta disser: filtro errado devolve zero resultados."""
        print(f"Retrieving calendar events for query: {query} (term={term}, campus={campus}, year={year})")
        retrievedDocs = self.rerank(
            query, self.eventRetriever.search(query, term=term, campus=campus, year=year)
        )
        print(f"Retrieved {len(retrievedDocs)} calendar documents.")
        if not retrievedDocs:
            return "Nenhum resultado encontrado.", []

        return self.serialize(retrievedDocs), retrievedDocs


    def queryOrRespond(self, state: MyState):
        llmWithTools = self.llm.bind_tools(self.tools)
        systemMessageContent = """Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da faculdade UTFPR.

        Você tem duas ferramentas de busca:
        - searchRegulations: regras, direitos, deveres, critérios e prazos previstos em regulamentos e normas.
        - searchCalendar: datas e eventos do calendário acadêmico.

        Sempre use as ferramentas antes de responder e use quantas forem necessárias, uma por vez, até reunir toda a informação.
        Se a pergunta tiver mais de um assunto, faça uma busca para cada assunto.
        Quando os resultados já responderem a pergunta, não busque de novo.
        A entrada é no formato "query": "frase para busca RAG", apenas a string, sem type ou outras coisas."""
        response = llmWithTools.invoke(
            [SystemMessage(content=systemMessageContent)] + state["messages"]
        )
        print(response)
        print(response.tool_calls)
        return {"messages": [response], "rounds": state.get("rounds", 0) + 1}

    def shouldContinue(self, state: MyState):

        lastMessage = state["messages"][-1]

        if not getattr(lastMessage, "tool_calls", None):
            return "generate"

        if state.get("rounds", 0) > MAX_ROUNDS:
            print(f"Limite de {MAX_ROUNDS} rodadas atingido, gerando a resposta.")
            return "generate"

        return "tools"



    async def generate(self, state: MyState):

        docs = self.collectDocs(state["messages"])

        docsContent = "\n\n".join(
            f"[{i}] Fonte: {self.docLabel(doc)}\nConteudo: {doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        )

        candidates = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]

        yield {"candidates": candidates}

        # systemMessageContent = (
        #     """Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da UTFPR.
        #     Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
        #     Cite a fonte quando fornecer a informação, nunca altere o link. Se não souber a resposta ou não haver documentos, diga que não sabe.
        #     Sempre escreva no formato markdown


        #     Documentos:
        #     \n\n
        #     """
        #     f"{docsContent}"


        # )
        systemMessageContent = ("""
        Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da UTFPR.
        Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
        Cada documento está numerado. Cite a fonte de cada afirmação com o marcador [n] correspondente, logo depois da informação.
        Nunca cite um documento que você não usou e nunca invente números.
        Se não souber a resposta ou não haver documentos, diga que não sabe.
        Sempre escreva no formato markdown


        Documentos:
        \n\n
        """
        f"{docsContent}"
        )


        messages = state["messages"]

        while (
            messages
            and messages[-1].type == "ai"
            and not getattr(messages[-1], "tool_calls", None)
        ):
            messages = messages[:-1]

        conversationMessages = [
            message
            for message in messages
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(systemMessageContent)] + conversationMessages

        async for chunk in self.llm.astream(prompt):
            yield {
                "messages": [chunk]
            }



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

    def saveMessage(self, threadId, role, content, sources = []):

        query = """
        MERGE (c:Chat {thread_id:$thread_id})

        CREATE (m:Message {
            id: randomUUID(),
            role: $role,
            content: $content,
            sources: $sources,
            timestamp: datetime()
        })

        MERGE (c)-[:HAS_MESSAGE]->(m)
        """

        self.db.executeQuery(
            query, parameters = {
                "thread_id": threadId,
                "role": role,
                "content": content,
                "sources": sources
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

        cited = []
        for number in re.findall(r"\[(\d+)\]", answer or ""):
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

    async def run(self, message, sessionId, persist=True):

        history = self.getRecentMessages(sessionId) if persist else []

        stateMessages = history + [
            {"role": "user", "content": message}
        ]

        if persist:
            self.saveMessage(sessionId, "user", message)

        fullAnswer = ""
        candidates = []
        retrieved = []
        seen = set()

        async for event in self.graph.astream_events(
            {"messages": stateMessages, "rounds": 0},
            version="v1",
        ):

            eventType = event["event"]
            node = event.get("metadata", {}).get("langgraph_node")

            if eventType == "on_chat_model_stream" and node in ("generate", None):
                chunk = event["data"].get("chunk")
                content = getattr(chunk, "content", None)

                if content:
                    fullAnswer += content
                    yield ("token", content)

            elif eventType == "on_chain_stream" and node == "generate":
                chunk = event["data"].get("chunk")

                if isinstance(chunk, dict) and chunk.get("candidates"):
                    candidates = chunk["candidates"]

            elif eventType == "on_tool_end":
                toolMessage = event["data"].get("output")
                documents = getattr(toolMessage, "artifact", None) or []

                for doc in documents:
                    key = doc.metadata.get("id") or doc.page_content
                    if key in seen:
                        continue
                    seen.add(key)
                    retrieved.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    })

                yield ("searching", {
                    "tool": event.get("name"),
                    "encontrados": len(documents),
                })

            await asyncio.sleep(0)

        if not candidates:
            candidates = retrieved

        sources = await self.usedSources(fullAnswer, candidates)

        if persist:
            self.saveMessage(sessionId, "assistant", fullAnswer, sources=json.dumps(sources))

        yield ("contexts", candidates)

        yield ("sources", sources)

    async def answer(self, message, chatHistory, sessionId, persist=True):

        fullAnswer = ""
        sources = []
        contexts = []

        async for kind, payload in self.run(message, sessionId, persist=persist):
            if kind == "token":
                fullAnswer += payload
            elif kind == "contexts":
                contexts = payload
            elif kind == "sources":
                sources = payload

        return {
            "answer": fullAnswer,
            "sources": sources,
            "contexts": contexts
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









    # def fullAnswer(self, message, chatHistory, sessionId):
    #     config = {"configurable": {"thread_id": sessionId}}

    #     result = self.graph.invoke(
    #         {"messages": [{"role": "user", "content": message}]},
    #         stream_mode="values",
    #         config=config,
    #     )

    #     resultAnswer = result["messages"]
    #     return resultAnswer
