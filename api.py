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
api = FastAPI()

@api.post("/rag")
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

api = gr.mount_gradio_app(api, interface, path="/chat")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
