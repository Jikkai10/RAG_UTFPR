from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


llm = ChatOllama(model="llama3.1", temperature=0.5)
model_name = "Alibaba-NLP/gte-multilingual-base"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, 
   
    model_kwargs={'trust_remote_code': True}
    
)
chromadb_path = "./db" # CONFIG YOUR PATH
vector_store = Chroma(
    #client=chroma_client,
    collection_name="rag",
    embedding_function=embeddings,
    persist_directory=chromadb_path
)
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Você é um assistente baseado em um modelo de linguagem de IA.
    Sua tarefa é gerar três versões diferentes da pergunta feita pelo usuário para recuperar documentos relevantes de um banco de dados vetorial.
    Ao gerar múltiplas perspectivas da pergunta original, seu objetivo é ajudar o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
    Forneça essas perguntas alternativas separadas por quebras de linha. Retorne apenas as perguntas, sem explicações adicionais ou coisas como 'aqui estão as perguntas'.
    Pergunta original: {question}""",
)


llm_retriever = ChatOllama(model="llama3.1", temperature=0)
llm_chain = QUERY_PROMPT | llm_retriever | output_parser
retriever = MultiQueryRetriever(
    retriever=vector_store.as_retriever(), llm_chain=llm_chain, parser_key="lines", include_original=True
)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)



graph_builder = StateGraph(MessagesState)
def get_embedding(text):

    embedding = embeddings.embed_query(text)

    return embedding

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    #retrieved_docs = vector_store.similarity_search(query, k=5)
    retrieved_docs = compression_retriever.invoke(query)

    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
    )

    return serialized, retrieved_docs
    #return documents_str, relevant_documents

def query_or_respond(state: MessagesState):
    """Generate tool call retrieve or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate a answer."""

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

    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

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

graph = graph_builder.compile()

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

input_message = "qual o prazo máximo de entrega para o relatório parcial de estágio?"
from rich.console import Console
from rich.markdown import Markdown
console = Console()
import gradio as gr
import uuid
def responder(mensagem, chat_history, session_id):
    config = {"configurable": {"thread_id": session_id}}
    
    result = graph.invoke(
        {"messages": [{"role": "user", "content": mensagem}]},
        stream_mode="values",
        config=config,
    )

    resposta = result["messages"][-1].content
    return resposta
    


with gr.Blocks() as interface:
    session_id_state = gr.State(str(uuid.uuid4()))  # gera um session_id aleatório

    gr.Markdown("# 🤖 Chat RAG")

    gr.ChatInterface(
        responder,
        type="messages",
        chatbot=gr.Chatbot(height="60vh"),
        additional_inputs=[
            session_id_state
        ],
    )


interface.launch()



# input_message = "Quais sao as possiveis modalidades de ensino?"


# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     md = Markdown(step["messages"][-1].content)
#     console.print(md)
#     console.print("--" * 20)

        