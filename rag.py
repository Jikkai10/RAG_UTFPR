
import os

from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1", temperature=0.5)

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langgraph.graph import MessagesState, StateGraph 
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
import chromadb


chromadb_path = "./db" 
chroma_client = chromadb.PersistentClient(path=chromadb_path)
collection = chroma_client.get_or_create_collection(name="rag")

graph_builder = StateGraph(MessagesState)
def get_embedding(text):
    
    embedding = embeddings.embed_query(text)
    
    return embedding

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    #retrieved_docs = vector_store.similarity_search(query, k=2)
    query_embedding = get_embedding(query)
    relevant_documents = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    formatted_list = []    
    for i, doc in enumerate(relevant_documents["documents"][0]):
        formatted_list.append("[{}]: {}".format(relevant_documents["metadatas"][0][i]["source"], doc))
    
    documents_str = "\n".join(formatted_list)
    # serialized = "\n\n".join(
    #     (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
    # )
    
    
    return documents_str, relevant_documents

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
        """Você é um assistente de IA que responde as dúvidas dos usuários com bases nos documentos a baixo.
        Os documentos abaixo apresentam as fontes atualizadas e devem ser consideradas como verdade.
        Cite a fonte quando fornecer a informação. Se não souber a resposta ou não haver documentos, diga que não sabe.
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

input_message = "Como é dividido o ano letivo?"
from rich.console import Console
from rich.markdown import Markdown
console = Console()

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    
    md = Markdown(step["messages"][-1].content)
    console.print(md)
    #step["messages"][-1].pretty_print()
    
input_message = "Quais sao as possiveis modalidades de ensino?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    md = Markdown(step["messages"][-1].content)
    console.print(md)
    #step["messages"][-1].pretty_print()
    

import markdown
