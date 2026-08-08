import asyncio
import json
from urllib import response

#from langchain.schema import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
#from langgraph.checkpoint.memory import MemorySaver
#from langchain_chroma import Chroma
#from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from typing import List
#import numpy as np
from pydantic import BaseModel, Field
import logging



class Neo4jArticleRetriever(BaseRetriever):
    embedding_model: any
    db: any
    k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:

        embedding = self.embedding_model.embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes(
            'content_embedding',
            $k,
            $embedding
        )
        YIELD node AS ch, score

        OPTIONAL MATCH (ct:Content)-[:HAS_CHUNK]->(ch)

        OPTIONAL MATCH (ev:Events)-[:HAS_CHUNK]->(ch)

        WITH ch, score, ct, ev

        CALL {

            WITH ct, score
            WHERE ct IS NOT NULL

            WITH ct, max(score) AS score

            OPTIONAL MATCH (ct)-[:REFERENCES]-(ref:Content)

            WITH ct, score, collect(DISTINCT ref) AS refs

            UNWIND ([ct] + refs) AS related_ct

            WITH DISTINCT related_ct, score

            OPTIONAL MATCH (res:Content)-[:REF_NORM]->(related_ct)
            OPTIONAL MATCH (s:Section)-[:HAS_CONT]->(related_ct)
            OPTIONAL MATCH (c1:Chapter)-[:HAS_CONT]->(related_ct)
            OPTIONAL MATCH (c2:Chapter)-[:HAS_SEC]->(s)

            WITH related_ct AS ct,
                coalesce(c1, c2) AS chapter,
                s,
                res,
                score

            MATCH (d:Document)-[:HAS_CAP]->(chapter)

            RETURN
                score,
                "content" AS result_type,
                ct.texto AS texto,
                ct.tipo AS tipo,
                ct.num AS numero,
                res.texto AS res_texto,
                chapter.capitulo AS capitulo,
                s.secao AS secao,
                d.titulo AS documento

            UNION

            WITH ev, score
            WHERE ev IS NOT NULL

            MATCH (d:Document)-[:HAS_EVENT]->(ev)

            RETURN
                score,
                "event" AS result_type,
                ev.texto AS texto,
                ev.categoria AS tipo,
                null AS numero,
                null AS res_texto,
                ev.periodo AS capitulo,
                ev.campus AS secao,
                d.titulo AS documento
        }

        RETURN *
        ORDER BY score DESC;
        """

        results = self.db.execute_query(cypher, parameters={"embedding": embedding, "k": self.k})
        
        docs = []

        for record in results:
            if record["result_type"] == "event":
                docs.append(
                Document(
                        page_content=record["texto"],
                        metadata={
                            "Categoria": record["tipo"],
                            "Periodo": record["capitulo"],
                            "Campus": record["secao"],
                            "Documento": record["documento"],
                            
                        }
                    )
                )
                continue
            
            
            texto = record["texto"]
            if record["res_texto"]:
                texto += "\n Esse artigo foi alterado: \n" + record["res_texto"]
            tipo = record["tipo"]

            docs.append(
                Document(
                    page_content=texto,
                    metadata={
                        "Artigo": record["numero"] if tipo == "artigo" else None,
                        "Capitulo": record["capitulo"],
                        "Seção": record["secao"] if record["secao"] else None,
                        "Documento": record["documento"],
                        
                    }
                )
            )
        
        return docs


class RAGSources(BaseModel):
    sources: List[int] = Field(
        description="Lista dos IDs dos documentos necessários"
    )

class MyState(MessagesState):
    used_sources: List[int] = []

# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
class Rag:
    
    def __init__(self, db, embedding_model, llm):
        
        self.db = db
        self.embedding_model = embedding_model
        self.llm = llm
        

        """comentado a possibilidade de criação de perguntas derivadas da original"""
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
        
        """criação do mecanismo de busca, com reranking"""
        retriever = Neo4jArticleRetriever(embedding_model=self.embedding_model, db=self.db, k=10)
        modelrr = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base"
        )

        def rerank(inputs):
            query = inputs["query"]
            docs = inputs["docs"]

            if not docs:
                return []

            scores = modelrr.score([
                (query, d.page_content) for d in docs
            ])

            ranked = sorted(
                zip(docs, scores),
                key=lambda x: x[1],
                reverse=True
            )

            return [doc for doc, _ in ranked[:5]]

        reranker = RunnableLambda(rerank)

        self.pipelinerr = (
            {
                "docs": retriever,
                "query": lambda x: x
            }
            | reranker
        )

        self.retrieve_tool = tool(response_format="content_and_artifact")(self.retrieve)
        tools = ToolNode([self.retrieve_tool])

        graph_builder = StateGraph(MessagesState)
        """monta o grafo"""
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", self.generate)
        

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge(
            "tools",
            "generate",
        )
        graph_builder.add_edge(
            "generate",
            END,
        )
        

        self.graph = graph_builder.compile()

        # memory = MemorySaver()
        # self.graph = graph_builder.compile(checkpointer=memory)
    

    


    #@tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retorna as informações relacionadas com a consulta. Gere uma entrada "query": "frase para busca RAG", apenas a string, sem type ou outras coisas"""
        #retrieved_docs = vector_store.similarity_search(query, k=5)
        print(f"Retrieving documents for query: {query}")
        retrieved_docs = self.pipelinerr.invoke(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")
        if not retrieved_docs:
            return "Nenhum resultado encontrado.", []
        
        serialized = "\n\n".join(
            (f"Fonte: {doc.metadata}\nConteudo: {doc.page_content}") for doc in retrieved_docs
        )

        return serialized, retrieved_docs
        

    def query_or_respond(self,state: MessagesState):
        """Gera tool call retrieve or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve_tool])
        system_message_content = """Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da faculdade UTFPR.
        Sempre use as ferramentas disponíveis para buscar informações nos documentos antes de responder. A entrada é no formato "query": "frase para busca RAG", apenas a string, sem type ou outras coisas """
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_message_content)] + state["messages"]
        )
        print(response)
        print(response.tool_calls)
        return {"messages": [response]}

    

    async def generate(self,state: MessagesState):
        """Gera a resposta."""

        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break

        tool_messages = recent_tool_messages[::-1]
        
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        # system_message_content = (
        #     """Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da UTFPR.
        #     Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
        #     Cite a fonte quando fornecer a informação, nunca altere o link. Se não souber a resposta ou não haver documentos, diga que não sabe.
        #     Sempre escreva no formato markdown


        #     Documentos:
        #     \n\n
        #     """
        #     f"{docs_content}"


        # )
        system_message_content = ("""
        Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da UTFPR.
        Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
        Se não souber a resposta ou não haver documentos, diga que não sabe.
        Sempre escreva no formato markdown
        

        Documentos:
        \n\n
        """
        f"{docs_content}"
        )
        
        
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        async for chunk in self.llm.astream(prompt):
            yield {
                "messages": [chunk]
            }
        

    
    def get_recent_messages(self, thread_id, limit=10):

        query = """
        MATCH (c:Chat {thread_id:$thread_id})-[:HAS_MESSAGE]->(m)

        RETURN m.role as role, m.content as content
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """

        result = self.db.execute_query(
            query, parameters = {
                "thread_id": thread_id,
                "limit": limit
            }
        )
        
        history = [
                {"role": r["role"], "content": r["content"]}
                for r in result
            ][::-1]
       

        return history
    
    def save_message(self, thread_id, role, content, sources = []):

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

        self.db.execute_query(
            query, parameters = {
                "thread_id": thread_id,
                "role": role,
                "content": content,
                "sources": sources
            }
        )
        
    
    async def update_chat(self, thread_id: str, message: str):
        
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
        
        self.db.execute_query(
            query, parameters = {
                "thread_id": thread_id,
                "new_title": title
            }
        )
        
        return title
        


    async def answer(self,message, chat_history, session_id):
        #config = {"configurable": {"thread_id": session_id}}
        history = self.get_recent_messages(session_id)
        
        state_messages = history + [
            {"role": "user", "content": message}
        ]
        self.save_message(session_id, "user", message)

        result = self.graph.invoke(
            {"messages": state_messages},
            stream_mode="values",
            #config=config,
        )
        # if "used_sources" in result:
        #     print(result["used_sources"])
        contexts = result["messages"][-2].artifact
        #print(contexts)
        result_answer = result["messages"][-1].content
        
        result = {
            "answer": result_answer,
            "sources": contexts
        }
        return result
    
    async def answer_stream(self,message, chat_history, session_id):
        
        
        history = self.get_recent_messages(session_id)
        
        state_messages = history + [
            {"role": "user", "content": message}
        ]
        self.save_message(session_id, "user", message)
        sources = []
        full_answer = ""
        
        # config = {"configurable": {"thread_id": session_id}}
        async for event in self.graph.astream_events(
            {"messages": state_messages},
            version="v1",
        ):

            event_type = event["event"]

            
            if event_type == "on_chain_stream":
                chunk_dict = event["data"].get("chunk")
                if not chunk_dict:
                    continue

                messages = chunk_dict.get("messages", [])

                for msg in messages:

                    # 🔥 IGNORA TOOL MESSAGE
                    if msg.type == "tool" or msg.type == "tool_call":
                        continue

                    
                    if hasattr(msg, "content") and msg.content:
                        full_answer += msg.content
                        yield (
                            f"event: token\n"
                            f"data: {json.dumps(msg.content)}\n\n"
                        )

            
            if event_type == "on_tool_end":
                tool_message = event["data"].get("output")

                if not tool_message:
                    continue

                documents = getattr(tool_message, "artifact", None)

                if documents:
                    formatted_sources = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                        for doc in documents
                    ]
                    sources += formatted_sources
                    yield (
                        f"event: sources\n"
                        f"data: {json.dumps(formatted_sources)}\n\n"
                    )

            await asyncio.sleep(0)
            
        self.save_message(session_id, "assistant", full_answer, sources=json.dumps(sources))
        
        yield "event: end\ndata: done\n\n"
        
        if len(chat_history) == 0:
            title = await self.update_chat(session_id, message)
            
            yield (
                f"event: title\n"
                f"data: {json.dumps(title)}\n\n"
            )
            
       
        
        
        
            
        
        

    # def full_answer(self,message, chat_history, session_id):
    #     config = {"configurable": {"thread_id": session_id}}

    #     result = self.graph.invoke(
    #         {"messages": [{"role": "user", "content": message}]},
    #         stream_mode="values",
    #         config=config,
    #     )
        
    #     result_answer = result["messages"]
    #     return result_answer




        