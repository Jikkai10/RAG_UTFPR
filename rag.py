import asyncio
import json

from langchain.schema import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
import numpy as np
from pydantic import BaseModel, Field
import logging


class Neo4jArticleRetriever(BaseRetriever):
    embedding_model: any
    db: any
    k: int = 20

    def _get_relevant_documents(self, query: str) -> List[Document]:

        embedding = self.embedding_model.embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes(
            'content_embedding',
            $k,
            $embedding
        )
        YIELD node AS ch, score

        MATCH (ct:Content)-[:HAS_CHUNK]->(ch)
        
        WITH ct, max(score) AS score

        OPTIONAL MATCH (s:Section)-[:HAS_CONT]->(ct)
        OPTIONAL MATCH (c1:Chapter)-[:HAS_CONT]->(ct)
        OPTIONAL MATCH (c2:Chapter)-[:HAS_SEC]->(s)

        WITH 
            ct,
            coalesce(c1, c2) AS chapter,
            s,
            score

        MATCH (d:Document)-[:HAS_CAP]->(chapter)

        RETURN
            score,
            ct.texto AS texto,
            ct.tipo AS tipo,
            ct.num as numero,
            chapter.capitulo AS capitulo,
            s.secao AS secao,
            d.titulo AS documento
        ORDER BY score DESC
        """

        results = self.db.execute_query(cypher, parameters={"embedding": embedding, "k": self.k})
        
        docs = []

        for record in results:
            texto = record["texto"]
            tipo = record["tipo"]

            docs.append(
                Document(
                    page_content=texto,
                    metadata={
                        "Artigo": record["numero"] if tipo == "artigo" else None,
                        "Capitulo": record["capitulo"],
                        "Seção": record["secao"] if record["secao"] else None,
                        "Documento": record["documento"],
                        "score": record["score"]
                    }
                )
            )
        #print(docs)
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
        retriever = Neo4jArticleRetriever(embedding_model=self.embedding_model, db=self.db, k=20)
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=5)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        retrieve_tool = tool(response_format="content_and_artifact")(self.retrieve)
        tools = ToolNode([retrieve_tool])

        graph_builder = StateGraph(MessagesState)
        """monta o grafo"""
        graph_builder.add_node(self.query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(self.generate)
        

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

        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)
    

    


    #@tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retorna as informações relacionadas com a consulta."""
        #retrieved_docs = vector_store.similarity_search(query, k=5)
        retrieved_docs = self.compression_retriever.invoke(query)

        
        serialized = "\n\n".join(
            (f"Fonte: {doc.metadata}\nConteudo: {doc.page_content}") for doc in retrieved_docs
        )

        return serialized, retrieved_docs
        

    def query_or_respond(self,state: MessagesState):
        """Gera tool call retrieve or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])

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

        #structured_llm = self.llm.with_structured_output(RAGResponse)
        # response = self.llm.invoke(prompt)
        
        # return  {
        # "messages": [
        #     response
        # ],
        
        # }
        async for chunk in self.llm.astream(prompt):
            yield {
                "messages": [chunk]
            }
        
    def sources_used(self, state: MyState):
        """Retorna os IDs dos documentos usados."""
        
        last_mensage = state["messages"][-1].content
        other_mensages = state["messages"][:-1]
        
        recent_tool_messages = []
        for message in reversed(other_mensages):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break

        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        
        system_message_content = ("""
        Você é um assistente de IA com o objetivo de identificar quais documentos foram utilizados para gerar a resposta fornecida. 
        A resposta foi gerada com base em uma série de documentos, cada um identificado por um ID específico. 
        Sua tarefa é comparar cuidadosamente a resposta com o conteúdo dos documentos fornecidos e identificar quais documentos foram explicitamente utilizados para gerar a resposta.
        Compare cuidadosamente a resposta com cada documento.
        Retorne apenas os IDs cujos textos contêm explicitamente as informações usadas na resposta.
        Se não houver correspondência clara, não retorne o ID.
        A resposta é a seguinte:
        \n\n
        """
        f"{last_mensage}"
        """
        \n\n
        Os documentos fornecidos foram:
        \n\n
        """
        f"{docs_content}"
        )
        prompt = [SystemMessage(system_message_content)]
        
        structured_llm = self.llm.with_structured_output(RAGSources)
        
        response = structured_llm.invoke(prompt)
        
        return {
            "used_sources": response.sources,
        }

    



    async def answer(self,message, chat_history, session_id):
        # config = {"configurable": {"thread_id": session_id}}

        # result = self.graph.invoke(
        #     {"messages": [{"role": "user", "content": message}]},
        #     stream_mode="values",
        #     config=config,
        # )
        # # if "used_sources" in result:
        # #     print(result["used_sources"])
        # contexts = result["messages"][-2].artifact
        # #print(contexts)
        # result_answer = result["messages"][-1].content
        
        # result = {
        #     "answer": result_answer,
        #     "sources": contexts
        # }
        # return result
        config = {"configurable": {"thread_id": session_id}}

        async for event in self.graph.astream_events(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
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

                    yield (
                        f"event: sources\n"
                        f"data: {json.dumps(formatted_sources)}\n\n"
                    )

            await asyncio.sleep(0)

        # 🔹 3️⃣ EVENTO FINAL
        yield "event: end\ndata: done\n\n"

    def full_answer(self,message, chat_history, session_id):
        config = {"configurable": {"thread_id": session_id}}

        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config=config,
        )
        
        result_answer = result["messages"]
        return result_answer




        