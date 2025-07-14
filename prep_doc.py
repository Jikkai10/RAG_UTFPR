from unstructured.partition.html import partition_html

from langchain_ollama import OllamaEmbeddings
import os
import uuid
import chromadb
from langchain_chroma import Chroma

from paddleocr import FormulaRecognition
from langchain_core.documents import Document

import base64
import numpy as np
import cv2 
import ollama
import re
from typing import List, Tuple, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
chromadb_path = "./db" # CONFIG YOUR PATH
chroma_client = chromadb.PersistentClient(path=chromadb_path)
collection = chroma_client.get_or_create_collection(name="rag")
vector_store = Chroma(
    client=chroma_client,
    collection_name="rag",
    embedding_function=embeddings,
)
output_path = "./data/"
#file_path = output_path + 'RODP2.pdf'

os.environ["TESSDATA_PREFIX"] = os.path.abspath('./tessdata')

model = FormulaRecognition(model_name="PP-FormulaNet_plus-M")

CAP_RE   = re.compile(r"^\s*Cap[ií]tulo\s+([IVXLCDM]+)\b.*",  re.IGNORECASE)
SEC_RE   = re.compile(r"^\s*Se[cç][ãa]o\s+([IVXLCDM]+)\b.*", re.IGNORECASE)
ART_RE   = re.compile(r"^\s*Art(?:igo)?\.\s*(\d+)[ºo]?\b.*", re.IGNORECASE)


def get_description(table):
    response = ollama.chat(
        model='llama3.1',
        messages=[{
            'role': 'user',
            'content': f"""
                    forneça uma descrição simples e precisa da tabela a seguir, não forneça mais nada, apenas a descrição
                    tabela:
                    {table} 
            """,
            
        }]
    )
    #print(response['message']['content'])
    
    return response['message']['content']


def maybe_split(long_text: str,
                capitulo: str,
                secao: str,
                artigo: str,
                nome: str,
                url: str) -> List[Tuple[str, Dict]]:
    """
    Se o artigo for muito grande, divide mantendo o cabeçalho.
    """
    header = f"Capítulo {capitulo}\nSeção {secao}\nArt. {artigo}\n\n"
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,  
        length_function=len,
    )
    texts = splitter.split_text(long_text)
    out = []
    for text in texts:
        out.append((header + text.strip(),
                    {"capitulo": capitulo, "secao": secao, "artigo": artigo, "fonte": nome, "fonte_url": url   }))
    return out
    


def custom_split_by_hierarchy(full_text: str,
                              nome: str,
                              url: str
                              ) -> Tuple[List[str], List[Dict[str, str]]]:
    
    current_cap = "—"   # travessão indica 'desconhecido' nos primeiros artigos
    current_sec = "—"
    
    chunks: List[str] = []
    metadatas: List[Dict[str, str]] = []

    
    buff_art_lines = []
    art_number = "-"
    
    def flush_article():
        nonlocal buff_art_lines, art_number, chunks, metadatas
        if art_number is None or not buff_art_lines:
            return
        article_text = "\n".join(buff_art_lines)
        result = maybe_split(article_text, current_cap, current_sec, art_number, nome, url)
        for text, meta in result:
            chunks.append(text)
            metadatas.append(meta)
        buff_art_lines = []
        art_number = None
    
    aux = "" 
    for el in full_text:
        
        if "Table" in str(type(el)):
            t = aux + el.metadata.text_as_html + "\n"
            desc = get_description(t)
            desc = desc + "\n" + t
            header = f"Capítulo {current_cap}\nSeção {current_sec}\n\n"
            chunks.append(header + desc.strip())
            metadatas.append({"capitulo": current_cap, "secao": current_sec, "artigo": "-", "fonte": nome, "fonte_url": url})
            aux = ""
            continue
        
        if "Image" in str(type(el)):
            img = el.metadata.image_path
            
            img_bytes = base64.b64decode(el.metadata.image_base64)  
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            output = model.predict(input=img, batch_size=1)
            text = ""
            for res in output:
                text += res['rec_formula'] + "\n"
            buff_art_lines.append(text)
            aux = ""
            continue
        
        if "Text" in str(type(el)):
            if (m := CAP_RE.match(el.text)):
                flush_article()
                current_cap = m.group(1)     # ex: 'I', 'II'…
                current_sec = "—"            # reinicia seção
                continue                      

            if (m := SEC_RE.match(el.text)):
                flush_article()
                current_sec = m.group(1)
                continue

            if (m := ART_RE.match(el.text)):
                flush_article()
                art_number = m.group(1)       # número do artigo
                
                buff_art_lines.append(el.text)
                continue
            
            aux = el.text + "\n"
            
            if art_number:
                buff_art_lines.append(el.text)
    
    # Final do arquivo: descarrega o último artigo
    flush_article()
    return chunks, metadatas

def get_document(doc):
    
    elements = partition_html(
        url=doc["url"],
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    
    return custom_split_by_hierarchy(elements, doc["name"], doc["url"] )



def insert_data(documents, metadatas):

    
    vector_store.add_documents(
        [
            Document(
                page_content=doc,
                metadata=metadatas[i],
                id=str(uuid.uuid4())
            )
            for i, doc in enumerate(documents)
        ]
    )
    print(f"Data successfully entered! {len(documents)} Chunks")

docs =[
    {
        "name": "RODP",
        "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1033898&id_orgao_publicacao=0",
    },
    {   
        "name": "SEI",
        "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1608522&id_orgao_publicacao=0",
    },
    
]

def run():
    print("Running prep docs...")

    documents = []
    metadatas = []

    for i, doc in enumerate(docs):
        print(f"Processing {i+1}/{len(docs)}: {doc['name']}")
        
        chunks, meta = get_document(doc)
        
        documents.extend(chunks)
        metadatas.extend(meta)

    
    insert_data(documents, metadatas)

    # documents_names = os.listdir(data_path)
    # documents_names_size = len(documents_names)
    # for i, document_name in enumerate(documents_names): 
    #     print(f"{i+1}/{documents_names_size}: {document_name}")

    #     document_texts, document_chunks = get_document(os.path.join(data_path, document_name))
    #     #document_embeddings, document_metadatas = prepare_documents(document_chunks, document_name)
    #     documents.extend(document_chunks)
    #     #embeddings.extend(document_embeddings)
    #     #metadatas.extend(document_metadatas)
    
    # ids = create_ids(documents)
    # insert_data(documents)
    
if __name__ == "__main__":
    run()
