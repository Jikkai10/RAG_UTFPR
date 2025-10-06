from typing import List
import gradio as gr
import uuid
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uuid
import rag
import prep_doc
import uvicorn

class MessageRequest(BaseModel):
    message: str
    chat_history: List[dict] = []
    session_id: str
    
class DocumentsPost(BaseModel):
    documents: List[dict]
    
app = FastAPI()

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
