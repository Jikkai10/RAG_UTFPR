from typing import List
import gradio as gr
import uuid
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uuid
import rag
import uvicorn

class MessageRequest(BaseModel):
    message: str
    chat_history: List[dict] = []
    session_id: str
app = FastAPI()

@app.post("/rag")
def answer_api(request: MessageRequest):
    response = rag.answer(request.message, request.chat_history, request.session_id)
    return response

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
