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
import logging


# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
class Rag:
    
    def __init__(self, vector_store, llm):
        
        self.vector_store = vector_store
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
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
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
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
        )

        return serialized, retrieved_docs
        

    def query_or_respond(self,state: MessagesState):
        """Gera tool call retrieve or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])

        return {"messages": [response]}

    

    def generate(self,state: MessagesState):
        """Gera a resposta."""

        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break

        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            """Você é um assistente de IA que responde as dúvidas dos usuários sobre os documentos oficiais da UTFPR.
            Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
            Cite a fonte quando fornecer a informação, nunca altere o link. Se não souber a resposta ou não haver documentos, diga que não sabe.
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

        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    



    def answer(self,message, chat_history, session_id):
        config = {"configurable": {"thread_id": session_id}}

        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config=config,
        )
        
        result_answer = result["messages"][-1].content
        return result_answer

    def full_answer(self,message, chat_history, session_id):
        config = {"configurable": {"thread_id": session_id}}

        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="values",
            config=config,
        )
        
        result_answer = result["messages"]
        return result_answer




        