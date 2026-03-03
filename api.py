from typing import List
import chromadb
import gradio as gr
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel
import uuid
from db.connection import Neo4jConnection
from rag import Rag
from extract_info.extract import PrepDocs
import requests
from config import UPLOAD_DIR
from extract_info.util import retrieve_all_documents, delete_document
import os


class MessageRequest(BaseModel):
    message: str
    
class DocumentsPost(BaseModel):
    name: str
    url: str
    
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

llm = ChatOllama(model="llama3.2", temperature=0.5, base_url=ollama_url)
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

db = Neo4jConnection()    
app = FastAPI()

rag = Rag(embedding_model=embeddings, llm=llm, db=db)
prep_doc = PrepDocs(llm=llm, embedding=embeddings)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-pdf/")
async def upload_pdf(
    nome: str = Form(...),          # campo texto
    file: UploadFile = File(...)    # arquivo
):

    # gera nome único
    filename = f"{uuid.uuid4()}.pdf"
    file_path = UPLOAD_DIR / filename

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)
        
    doc = {
        "name": nome,
        "filename": filename
    }
    
    prep_doc.run([doc], 1)
    

@app.get("/chats")
def new_chat():
    session_id = str(uuid.uuid4())
    return {"id": session_id}

@app.post("/rag/{session_id}")
async def answer_api(session_id: str, request: MessageRequest):
    async def event_generator():
        async for chunk in rag.answer(
            request.message,
            [],
            session_id,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    # response = rag.full_answer(request.message, [], session_id)
    # return response

@app.post("/docs")
def post_docs(req: DocumentsPost):
    response = requests.get(req.url)

    if response.status_code != 200:
        return {"error": "Não foi possível baixar"}

    filename = f"{uuid.uuid4()}.html"
    file_path = UPLOAD_DIR / filename

    # 🔥 Salva o HTML bruto
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    doc = {
        "name": req.name,
        "filename": filename,
        "url": req.url
    }
    prep_doc.run([doc], 0)
    
@app.get("/all_docs")
def get_all_docs():
    return retrieve_all_documents()

@app.delete("/delete_doc/{doc_id}")
def delete_doc(doc_id: str):
    delete_document(doc_id)
    return {"message": f"Documento {doc_id} deletado com sucesso."}
    


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
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": "Não foi possível baixar"}

    filename = f"{uuid.uuid4()}.html"
    file_path = UPLOAD_DIR / filename

    # 🔥 Salva o HTML bruto
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    docs = [
        {
            "name": name,
            "filename": filename,
            "url": url
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
