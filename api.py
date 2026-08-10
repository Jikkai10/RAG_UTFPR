import json
import mimetypes
from typing import List
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import uuid
from db.connection import Neo4jConnection
from rag import Rag
from extract_info.extract import PrepDocs
import requests
from config import UPLOAD_DIR
from extract_info.util import retrieveAllDocuments, deleteDocument, returnDocument
from security.security import Authenticator
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
    docType: int = Field(alias="doc_type")
    parentId: str = Field(alias="pai_id")

ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")

llm = ChatOllama(model="llama3.2", temperature=0.5, base_url=ollamaUrl)
modelName = "intfloat/multilingual-e5-base"

embeddings = HuggingFaceEmbeddings(
    model_name=modelName,
    model_kwargs={'trust_remote_code': True}

)

db = Neo4jConnection()
auth = Authenticator()

rag = Rag(embeddingModel=embeddings, llm=llm, db=db)
prepDoc = PrepDocs(llm=llm, embedding=embeddings)

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



def getCurrentUser(token: HTTPAuthorizationCredentials = Depends(security)):

    payload = auth.decode(token.credentials)

    return payload

def adminRequired(user = Depends(getCurrentUser)):

    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    return user

@app.get("/download/{docId}")
async def download(docId: str, user = Depends(getCurrentUser)):
    doc = returnDocument(db, docId)
    path = UPLOAD_DIR / doc["path"]
    name = doc["titulo"].replace(" ", "_")

    ext = doc["path"].split('.')[-1]
    mediaType, _ = mimetypes.guess_type(path)
    return FileResponse(
        path,
        media_type=mediaType,
        filename= f'{name}.{ext}'
    )


@app.post("/upload-pdf/")
async def uploadPdf(
    name: str = Form(...),
    docType: int = Form(..., alias="doc_type"),
    parentId: str = Form(..., alias="pai_id"),
    file: UploadFile = File(...),
    user = Depends(adminRequired)
):

    # gera nome único
    filename = f"{uuid.uuid4()}.pdf"
    filePath = UPLOAD_DIR / filename

    contents = await file.read()

    with open(filePath, "wb") as f:
        f.write(contents)

    if(docType == 3):
        doc = {
            "name": name,
            "filename": filename,
        }
        prepDoc.getCalendarDocument(doc)
        return

    doc = {
        "name": name,
        "filename": filename,
        "doc_id": parentId
    }

    prepDoc.getPdfDocument(doc, docType)


@app.get("/create_chat")
def newChat(user = Depends(getCurrentUser)):
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

    sessionId = str(uuid.uuid4())

    db.executeQuery(query, parameters= {
        "thread_id": sessionId,
        "user_id": user["sub"],
        "title": "Novo chat"
    })

    return {"id": sessionId, "title": "Novo chat"}

@app.get("/chat")
def getChats(user = Depends(getCurrentUser)):
    query="""
    MATCH (u:User {id: $user_id})

    MATCH (u)-[:HAS_CHAT]->(c:Chat)

    RETURN c.thread_id as id, c.title as title
    ORDER BY c.create_at DESC
    """
    result = db.executeQuery(query, parameters= {
        "user_id": user["sub"],
    })


    return result

@app.delete("/chat/{threadId}")
def deleteChat(threadId: str, user = Depends(getCurrentUser)):
    query="""
    MATCH (c:Chat {thread_id: $thread_id, user_id: $user_id})
    OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
    DETACH DELETE c, m
    """

    db.executeQuery(
        query, parameters = {
            "thread_id": threadId,
            "user_id": user["sub"]
        }
    )

@app.put("/chat/{threadId}/{newTitle}")
def updateChat(threadId: str, newTitle: str, user = Depends(getCurrentUser)):
    query="""
    MATCH (c:Chat {thread_id: $thread_id, user_id: $user_id})
    SET c.title = $new_title
    RETURN c
    """

    db.executeQuery(
        query, parameters = {
            "thread_id": threadId,
            "user_id": user["sub"],
            "new_title": newTitle
        }
    )


@app.get("/chat/{threadId}")
def getHistory(threadId: str, user = Depends(getCurrentUser)):
    query = """
    MATCH (c:Chat {thread_id:$thread_id, user_id:$user_id})-[:HAS_MESSAGE]->(m)

    RETURN m.role as role, m.content as content, m.sources as sources
    ORDER BY m.timestamp DESC
    """

    result = db.executeQuery(
        query, parameters = {
            "thread_id": threadId,
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


@app.post("/rag/stream/{sessionId}")
async def answerStreamApi(sessionId: str, request: MessageRequest, user = Depends(getCurrentUser)):
    chatHistory = rag.getRecentMessages(sessionId)

    async def eventGenerator():
        async for chunk in rag.answerStream(
            request.message,
            chatHistory,
            sessionId,
        ):
            yield chunk

    return StreamingResponse(
        eventGenerator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    # response = rag.fullAnswer(request.message, [], sessionId)
    # return response

@app.post("/rag/{sessionId}")
async def answerApi(sessionId: str, request: MessageRequest, user = Depends(getCurrentUser)):
    response = await rag.answer(request.message, [], sessionId)

    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }

@app.post("/rag/eval/{sessionId}")
async def answerEvalApi(sessionId: str, request: MessageRequest, user = Depends(getCurrentUser)):
    return await rag.answer(request.message, [], sessionId, persist=False)

@app.post("/docs")
def postDocs(req: DocumentsPost, user = Depends(adminRequired)):
    response = requests.get(req.url)

    if response.status_code != 200:
        return {"error": "Não foi possível baixar"}

    filename = f"{uuid.uuid4()}.html"
    filePath = UPLOAD_DIR / filename

    with open(filePath, "w", encoding="utf-8") as f:
        f.write(response.text)

    doc = {
        "name": req.name,
        "filename": filename,
        "url": req.url,
        "doc_id": req.parentId
    }
    prepDoc.getDocument(doc, req.docType)

@app.get("/all_docs")
def getAllDocs(user = Depends(getCurrentUser)):
    return retrieveAllDocuments(db)

@app.delete("/docs/{docId}")
def deleteDoc(docId: str, user = Depends(adminRequired)):
    paths = deleteDocument(db, docId)

    for path in paths:
        filePath = UPLOAD_DIR / path
        if filePath:
            os.remove(filePath)
    return {"message": f"Documento {docId} deletado com sucesso."}


@app.post("/register")
def register(req: UserRequest):

    hashed = auth.hashPassword(req.password)
    query = """
        CREATE (u:User {
            id: $id,
            email: $email,
            password: $password,
            role: $role
        })
        """

    db.executeQuery(query, parameters={
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

    record = db.executeQuery(query, parameters={"email": req.email})
    if record:
        record = record[0]

    if not record:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    if not auth.verifyPassword(req.password, record["password"]):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    token = auth.createAccessToken({
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
def me(user = Depends(getCurrentUser)):
    return {
        "email": user["email"],
        "role": user["role"]
    }
