import gradio as gr
import uuid
import requests

API_URL = "http://localhost:8000/rag"

def responder(message, chat_history, session_id):
    payload = {
        "message": message,
        "chat_history": chat_history,
        "session_id": session_id
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return chat_history + [{"type": "system", "content": "Erro ao conectar com o servidor."}]   

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