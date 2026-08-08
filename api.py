import json
import mimetypes
from typing import List
import chromadb
import gradio as gr
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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
from extract_info.util import retrieve_all_documents, delete_document, return_document
from security.security import Autentify
import os
import logging

logger = logging.getLogger("uvicorn.error")


class MessageRequest(BaseModel):
    message: str
    
class UserRequest(BaseModel):
    password: str
    email: str
    
class DocumentsPost(BaseModel):
    name: str
    url: str
    doc_type: int
    pai_id: str
    
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
model_name = "intfloat/multilingual-e5-base"

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
sec = Autentify()

rag = Rag(embedding_model=embeddings, llm=llm, db=db)
prep_doc = PrepDocs(llm=llm, embedding=embeddings)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()



senha = sec.hash_password("senha")
query = """MERGE (u:User {id: $id})
        SET u.email = $email,
        u.password = $password,
        u.role = $role
"""

db.execute_query(query, parameters={
    "id": "1",
    "email": "admin@email.com",
    "password": senha,
    "role": "admin"
})

db.execute_query(query, parameters={
    "id": "2",
    "email": "user@email.com",
    "password": senha,
    "role": "user"
})

def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
    
    payload = sec.decode(token.credentials)

    return payload

def admin_required(user = Depends(get_current_user)):

    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    return user

@app.get("/download/{doc_id}")
async def download(doc_id: str):
    doc = return_document(db, doc_id)
    path = UPLOAD_DIR / doc["path"]
    name = doc["titulo"].replace(" ", "_")
    
    ext = doc["path"].split('.')[-1]
    media_type, _ = mimetypes.guess_type(path)
    return FileResponse(
        path,
        media_type=media_type,
        filename= f'{name}.{ext}'
    )


@app.post("/upload-pdf/")
async def upload_pdf(
    name: str = Form(...),          
    doc_type: int = Form(...),
    pai_id: str = Form(...),
    file: UploadFile = File(...),
    #user = Depends(admin_required) 
):

    # gera nome único
    filename = f"{uuid.uuid4()}.pdf"
    file_path = UPLOAD_DIR / filename

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)
        
    if(doc_type == 3):
        doc = {
            "name": name,
            "filename": filename,
        }  
        prep_doc.get_calendar_document(doc)
        return
    
    doc = {
        "name": name,
        "filename": filename,
        "doc_id": pai_id
    }
    
    prep_doc.get_pdf_document(doc, doc_type)
    

@app.get("/create_chat")
def new_chat(user = Depends(get_current_user)):
    query = """
    
    MATCH (u:User {id: $user_id})
    
    CREATE (c:Chat {
        thread_id: $thread_id,
        user_id: $user_id,
        title: $title,
        create_at: datetime()  
    })
    
    
    MERGE (u)-[:HAS_CHAT]->(c)
    """
    
    session_id = str(uuid.uuid4())
    
    db.execute_query(query, parameters= {
        "thread_id": session_id,
        "user_id": user["sub"],
        "title": "Novo chat"
    })
    
    return {"id": session_id, "title": "Novo chat"}

@app.get("/chat")
def get_chats(user = Depends(get_current_user)):
    query="""
    MATCH (u:User {id: $user_id})
    
    MATCH (u)-[:HAS_CHAT]->(c:Chat)
    
    RETURN c.thread_id as id, c.title as title
    ORDER BY c.create_at DESC
    """
    result = db.execute_query(query, parameters= {
        "user_id": user["sub"],
    })
    
    
    return result

@app.delete("/chat/{thread_id}")
def delete_chat(thread_id: str, user = Depends(get_current_user)):
    query="""
    MATCH (c:Chat {thread_id: $thread_id, user_id: $user_id})
    OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
    DETACH DELETE c, m
    """
    
    db.execute_query(
        query, parameters = {
            "thread_id": thread_id,
            "user_id": user["sub"]
        }
    )
    
@app.put("/chat/{thread_id}/{new_title}")
def update_chat(thread_id: str, new_title: str, user = Depends(get_current_user)):
    query="""
    MATCH (c:Chat {thread_id: $thread_id, user_id: $user_id})
    SET c.title = $new_title
    RETURN c
    """
    
    db.execute_query(
        query, parameters = {
            "thread_id": thread_id,
            "user_id": user["sub"],
            "new_title": new_title
        }
    )
    

@app.get("/chat/{thread_id}")
def get_history(thread_id: str, user = Depends(get_current_user)):
    query = """
    MATCH (c:Chat {thread_id:$thread_id, user_id:$user_id})-[:HAS_MESSAGE]->(m)

    RETURN m.role as role, m.content as content, m.sources as sources
    ORDER BY m.timestamp DESC
    """

    result = db.execute_query(
        query, parameters = {
            "thread_id": thread_id,
            "user_id": user["sub"]
        }
    )
    
    for record in result:
        if record["sources"]:
            try:
                record["sources"] = json.loads(record["sources"])
            except:
                record["sources"] = None
        
        if len(record["sources"]) == 0:
            record["sources"] = None
    
    
    
    return result[::-1]
    

@app.post("/rag/stream/{session_id}")
async def answer_api(session_id: str, request: MessageRequest, user = Depends(get_current_user)):
    async def event_generator():
        async for chunk in rag.answer_stream(
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

@app.post("/rag/{session_id}")
async def answer_api(session_id: str, request: MessageRequest, user = Depends(get_current_user)):
    response = rag.answer(request.message, [], session_id)
    return response

@app.post("/docs")
def post_docs(req: DocumentsPost, user = Depends(admin_required)):
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
        "url": req.url,
        "doc_id": req.pai_id
    }
    prep_doc.get_document(doc, req.doc_type)
    
@app.get("/all_docs")
def get_all_docs():
    return retrieve_all_documents(db)

@app.delete("/docs/{doc_id}")
def delete_doc(doc_id: str, user = Depends(admin_required)):
    paths = delete_document(db, doc_id)
    
    for path in paths:
        file_path = UPLOAD_DIR / path
        if file_path:
            os.remove(file_path)
    return {"message": f"Documento {doc_id} deletado com sucesso."}
    

@app.post("/register")
def register(req: UserRequest):

    hashed = sec.hash_password(req.password)
    query = """
        CREATE (u:User {
            id: $id,
            email: $email,
            password: $password,
            role: $role
        })
        """
    
    db.execute_query(query, parameters={
            "id": str(uuid.uuid4()),
            "email": req.email,
            "password": hashed,
            "role": "user"
        })

    return {"msg": "User created"}

@app.post("/login")
def login(req: UserRequest):
    query = """
        MATCH (u:User {email:$email})
        RETURN u.password AS password, u.id AS id, u.role AS role, u.email as email
        """
        
    record = db.execute_query(query, parameters={"email": req.email})
    if record:
        record = record[0]
    
    if not record:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    if not sec.verify_password(req.password, record["password"]):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = sec.create_access_token({
        "sub": str(record["id"]),
        "role": record["role"],
        "email": record["email"]
    })

    return {
        "access_token": token,
        "user": {
            "email": record["email"],
            "role": record["role"]
        }
    }
    
@app.get("/me")
def me(user = Depends(get_current_user)):
    return {
        "email": user["email"],
        "role": user["role"]
    }
