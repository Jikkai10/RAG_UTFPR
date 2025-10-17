from typing import List
import chromadb
import gradio as gr
import uuid
from fastapi import FastAPI, Request
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel
import uuid
from rag import Rag
from prep_doc import PrepDocs
import requests
import os


class MessageRequest(BaseModel):
    message: str
    chat_history: List[dict] = []
    session_id: str
    
class DocumentsPost(BaseModel):
    documents: List[dict]
    
ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

chroma_url = os.getenv("CHROMA_URL", "http://localhost:8000")
def get_chroma_client(host="http://localhost:8000"):
    try:
        # Verifica se o servidor está respondendo
        response = requests.get(f"{host}/api/v1/heartbeat", timeout=2)
        if response.status_code == 200:
            return chromadb.HttpClient(host=host)
    except requests.exceptions.RequestException:
        pass

    
    return None

client_chroma = get_chroma_client(chroma_url)

llm = ChatOllama(model="llama3.1", temperature=0.5, base_url=ollama_url)
model_name = "Alibaba-NLP/gte-multilingual-base"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, 
    model_kwargs={'trust_remote_code': True}
    
)
chromadb_path = "./data" 
vector_store = Chroma(
    client=client_chroma,
    collection_name="rag",
    embedding_function=embeddings,
    persist_directory=chromadb_path
)    

    
app = FastAPI()

rag = Rag(vector_store=vector_store, llm=llm)
prep_doc = PrepDocs(vector_store=vector_store, llm=llm)

@app.post("/rag")
def answer_api(request: MessageRequest):
    response = rag.answer(request.message, request.chat_history, request.session_id)
    return response

@app.post("/docs")
def post_docs(request: DocumentsPost):
    prep_doc.run(request.documents)
    

with gr.Blocks() as interface:
    session_id_state = gr.State(str(uuid.uuid4()))  # gera um session_id aleatório

    gr.Markdown("# 🤖 Chat RAG")

    gr.ChatInterface(
        rag.answer,
        type="messages",
        chatbot=gr.Chatbot(height="60vh"),
        additional_inputs=[
            session_id_state
        ],
    )

app = gr.mount_gradio_app(app, interface, path="/chat")

def process_data(name, url):
    docs = [
        {
            "name": name,
            "url": url,
        }
    ]
    prep_doc.run(docs)

with gr.Blocks() as process:
    with gr.Row():
        name = gr.Textbox(label="Nome do documento")
        url = gr.Textbox(label="url")
    
    
    output_message = gr.Markdown(visible=False)
    
    # Botão de submit
    submit_btn = gr.Button("Enviar")

    # Evento de clique
    submit_btn.click(
        fn=lambda: gr.Markdown(value="🔄 Processando...", visible=True),
        outputs=output_message
    ).then(
        fn=process_data,
        inputs=[name, url]
    ).then(
        fn=lambda: gr.Markdown(value="✅ Sucesso! Dados processados.", visible=True),
        outputs=output_message
    )
    
app = gr.mount_gradio_app(app, process, path="/docs_ui")
