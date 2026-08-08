from langchain_ollama import ChatOllama
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
import uuid
from langchain_core.messages import SystemMessage
from langchain_chroma import Chroma
from paddlex import create_model
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import base64
import numpy as np
import cv2 
import re
from typing import List, Tuple, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import chromadb
import pandas as pd
from io import StringIO
from extract_info.util import dict_to_list, inserir_estrutura, insert_calendar
from config import UPLOAD_DIR
from db.connection import Neo4jConnection
import fitz  # PyMuPDF

CAP_RE   = re.compile(r"^\s*Cap[ií]tulo\s+([IVXLCDM]+)\b.*",  re.IGNORECASE)
SEC_RE   = re.compile(r"^\s*Se[cç][ãa]o\s+([IVXLCDM]+)\b.*", re.IGNORECASE)
ART_RE   = re.compile(r"^\s*Art(?:igo)?\.?\s*(\d+)[ºo]?\b.*", re.IGNORECASE)
REF_RE = re.compile(r"\bArt(?:igo)?\.?\s*(\d+)[ºo]?\b", re.IGNORECASE)

def is_table_empty(table):
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        for cell in cells:
            text = cell.get_text(strip=True)
            if text:  # se houver qualquer conteúdo não vazio
                return False
    return True 

def get_table_text(table):
    df = pd.read_html(StringIO(table))[0]
    # o cabeçalho está ficando na primeira linha
    df.columns = df.iloc[0]
    df = df[1:]
    mark = df.to_markdown(index=False)
    
    return mark

class PrepDocs:
    def __init__(self, llm, embedding):
        self.llm = llm
        self.model = create_model("PP-FormulaNet_plus-M")
        self.embedding = embedding
        self.db = Neo4jConnection()

    def chunks_to_embeddings(self, chunk):
        return self.embedding.embed_query(chunk)

    def get_description(self,table):
        prompt = (
            """
                forneça uma descrição simples e precisa da tabela a seguir em até 1000 caracteres, não forneça mais nada, apenas a descrição
            """
            
        )
        response = self.llm.invoke([SystemMessage(prompt)] + [f"{table}"])
        
        return response.content


    def extract_info(self, long_text: str,
                    capitulo: str,
                    secao: str,
                    artigo: str,
                    nome: str,
                    url: str) -> List[Tuple[str, Dict]]:
        
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,  
            length_function=len,
        )
        texts = splitter.split_text(long_text)
        refs = []
        refs_re = REF_RE.finditer(long_text)
        for ref in refs_re:
            refs.append(ref.group(1))
            
        if refs:
            refs = refs[1:]
            
        out = {
            "doc_id": nome,
            "cap_id": capitulo,
            "sec_id": secao,
            "cont_id": artigo,
            "text": long_text,
            "chunk": texts,
            "refs": refs,
            "path": url,
            "type": "artigo"
        }
        
        
        return out
     
     
    def organize_chunks_by_hierarchy(self, doc, info):
        
        doc_id = info["doc_id"]
        cap_id = info["cap_id"]
        sec_id = info["sec_id"]
        cont_id = info["cont_id"]
        
        
        # Capítulo
        cap = doc["capitulos"].setdefault(cap_id, {
            "id": f"{doc['id']}_cap{cap_id}",
            "capitulo": cap_id,
            "secoes": {},
            "conteudos": {}
        })

        if sec_id != "—":  
            sec = cap["secoes"].setdefault(sec_id, {
                "id": f"{cap['id']}_sec{sec_id}",
                "secao": sec_id,
                "conteudos": {}
            })
        else:
            sec = cap

        
        art = sec["conteudos"].setdefault(cont_id, {
            "id": f"{sec['id']}_cont{cont_id}",
            "cont_num": cont_id,
            "tipo": info["type"],
            "texto": info["text"],
            "refs": info["refs"],
            "chunks": []
        })

        # Chunk
        for i, chunk in enumerate(info["chunk"]):
            chunk_id = f"{art['id']}_chunk{i}"
            art["chunks"].append({
                "id": chunk_id,
                "texto": chunk
            })
     
        

    def custom_split_by_hierarchy(self,full_text: str,
                                nome: str,
                                url: str,
                                tables: List[str] = None,
                                ) -> Tuple[List[str], List[Dict[str, str]]]:
        
        current_cap = "—"   # travessão indica 'desconhecido' nos primeiros artigos
        current_sec = "—"
        
        chunks: List[str] = []
        metadatas: List[Dict[str, str]] = []
        docs = {
            "id": str(uuid.uuid4()),
            "titulo": nome,
            "path": url,
            "capitulos": {}
        }
        
        buff_art_lines = []
        art_number = "-"
        
        def flush_article():
            nonlocal buff_art_lines, art_number, chunks, metadatas
            if art_number is None or not buff_art_lines:
                return
            article_text = "\n".join(buff_art_lines)
            info = self.extract_info(article_text, current_cap, current_sec, art_number, nome, url)
            self.organize_chunks_by_hierarchy(docs, info)
            buff_art_lines = []
            art_number = None
        
        aux = "" 
        cont_table = 0  
        for el in full_text:
            
            if "Table" in str(type(el)):
                
                if tables is None or cont_table >= len(tables):
                    table = get_table_text(el.metadata.text_as_html)
                else:
                    table = tables[cont_table]
                if table is None:
                    continue
                t = aux + table + "\n"

                desc = self.get_description(t)
                
                
                cont_table += 1
                
                out = {
                    "doc_id": nome,
                    "cap_id": current_cap,
                    "sec_id": current_sec,
                    "cont_id": f"tab_{cont_table}",
                    "text": t,
                    "chunk": [desc],
                    "refs": [],
                    "path": url,
                    "type": "tabela"
                }
                
                self.organize_chunks_by_hierarchy(docs, out)
                
                continue
            
            if "Image" in str(type(el)):
                img = el.metadata.image_path
                
                img_bytes = base64.b64decode(el.metadata.image_base64)  
                buf = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                output = self.model.predict(input=img, batch_size=1)
                text = ""
                for res in output:
                    text += res['rec_formula']
                text = "$$"+text+"$$ \n"
                
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
        
        flush_article()
        return docs

    

    def get_document(self,doc, doc_type = 0):
        """
            partition html naturalmente exclui informações adicionais nas tags, como rowspan e colspan,
            o que pode ser problemático para descobrir o formato das tabelas,
            por isso o tratamento das tabelas é feito separadamente.
        """
        file_path = UPLOAD_DIR / doc["filename"]

        if not file_path.exists():
            raise FileNotFoundError("Arquivo não encontrado")

        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        
        soup = BeautifulSoup(html_content, 'lxml')
        tables = soup.find_all('table')
        tables = [str(table) for table in tables if not is_table_empty(table)] 
        frases = []
        for table in tables:
            frases.append(get_table_text(table))
            
        elements = partition_html(
            filename=str(file_path),
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            
        )
        
        docs = self.custom_split_by_hierarchy(elements, doc["name"], doc["filename"], frases)
        
        list = dict_to_list(docs, doc_type, self.chunks_to_embeddings)
        
        if doc_type == 0:
            doc["doc_id"] = None
        inserir_estrutura(self.db, list, doc_type, doc["doc_id"])

    def get_pdf_document(self,doc, doc_type = 0):
        """
            inferencia de tabelas pode sair com erros
        """
        filepath = UPLOAD_DIR / doc["filename"]
        elements = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            languages=["por"],
            extract_images_in_pdf=True,
            include_page_breaks=False,
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )

        docs = self.custom_split_by_hierarchy(elements, doc["name"], doc["filename"])
        list = dict_to_list(docs, doc_type, self.chunks_to_embeddings)
        
        if doc_type == 0:
            doc["doc_id"] = None
        inserir_estrutura(self.db, list, doc_type, doc["doc_id"])



    def extract_layout(self,pdf_path):

        doc = fitz.open(pdf_path)

        blocks = []

        for page_num, page in enumerate(doc):

            page_blocks = page.get_text("blocks")

            for b in page_blocks:

                x0, y0, x1, y1, text, _, _ = b

                blocks.append({
                    "page": page_num,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "text": text.strip()
                })

        return blocks
    
    
    def layout_to_markdown(self,blocks):
        
        MESES = [
        "JANEIRO","FEVEREIRO","MARÇO","ABRIL","MAIO","JUNHO",
        "JULHO","AGOSTO","SETEMBRO","OUTUBRO","NOVEMBRO","DEZEMBRO"
        ]
        padrao = r'(\d{1,2}(?:\s*(?:a|e|,)\s*\d{1,2})*)\s+(.+?)(?=\s+\d{1,2}\s+[A-Za-zÁ-ÿ]|\Z)'

        md = []
        current_month = None
        actual_page = None
        cont = 0
        doc = []
        frag = {}
        mes = ""
        
        for b in blocks:

            text = b["text"].replace("\n", " ").strip()
            text = re.sub(r'_{3,}', '', text).strip()

            if actual_page != b["page"]:
                if md and current_month == None:
                    md = md[:-1]

                if md:
                    mes = "\n".join(md)
                    frag["chunks"].append({"texto": mes})
                    frag["md"] += f"\n{mes}"
                    doc.append(frag)

                md = []
                frag = {}
                frag["md"] = ""
                frag["chunks"] = []
                frag["periodo"] = "x"
                #frag["chunks"]["texto"] = []

                actual_page = b["page"]
                current_month = None
                md.append(f"\n# {b["text"]}")
                cont = 0
                continue
            actual_page = b["page"]

            if not text:
                continue

            # detectar mês
            if any(m in text for m in MESES):

                for m in MESES:
                    if m in text:
                        current_month = m
                        
                        mes = "\n".join(md)
                        frag["chunks"].append({"texto": mes})
                        frag["md"] += f"\n{mes}"
                        md = []
                        md.append(f"\n## {m}\n")
                        text = text.replace(m, "").strip()
                        break


            if re.match(padrao, text):
                for dia, descricao in re.findall(padrao, text):
                    md.append(f"- {dia} {descricao}")
                    #frag["chunks"].append({"texto": md[-1]})


            else:

                if current_month == None:
                    cont += 1
                if cont == 1:
                    frag["campus"] = text

                elif cont == 2:
                    frag["categoria"] = text

                else:
                    frag["periodo"] = text

                if md:
                    md[-1] += " " + text
                    
        mes = "\n".join(md)
        frag["chunks"].append({"texto": mes})
        frag["md"] += f"\n{mes}"
        doc.append(frag)
        #print(doc)
        return doc
    
    def get_calendar_document(self, doc):
        path = UPLOAD_DIR / doc["filename"]
        blocks = self.extract_layout(path)
        parts = self.layout_to_markdown(blocks)
        doc_id = str(uuid.uuid4())
        for part in parts:
            part["id"] = f"{doc_id}_{part["campus"]}_{part["categoria"]}_{part["periodo"]}"
            for i, chunk in enumerate(part["chunks"]):
                chunk["id"] = f"{part["id"]}_{i}"
                chunk["embedding"] = self.chunks_to_embeddings(chunk=chunk["texto"])
                
        docs = {
            "doc_id": doc_id,
            "titulo": doc["name"],
            "path": doc["filename"],
            "parts": parts
        }
        
        insert_calendar(self.db, docs)
        
        #docs = self.custom_split_by_hierarchy(elements, doc["name"], doc["filename"])
        # list = dict_to_list(docs, doc_type, self.chunks_to_embeddings)
        
        # if doc_type == 0:
        #     doc["doc_id"] = None
        # inserir_estrutura(self.db, list, doc_type, doc["doc_id"])
    

    def run(self,docs, mode=1):
        print("Running prep docs...")

        documents = []
        metadatas = []

        for i, doc in enumerate(docs):
            print(f"Processing {i+1}/{len(docs)}: {doc['name']}")
            if mode == 1:
                self.get_document(doc, 0)
            else:
                self.get_pdf_document(doc, 0)


    
            
    
if __name__ == "__main__":
    docs =[
        {
            "name": "REGULAMENTO DA ORGANIZAÇÃO DIDÁTICO-PEDAGÓGICA",
            "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1033898&id_orgao_publicacao=0",
        },
        {   
            "name": "REGULAMENTO DOS ESTÁGIOS CURRICULARES SUPERVISIONADOS",
            "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1608522&id_orgao_publicacao=0",
        },
        
        {   
            "name": "REGULAMENTO DE TRABALHO DE CONCLUSÃO DE CURSO",
            "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=3171226&id_orgao_publicacao=0",
        },
    
    ]
    
    # docs2=[
    #     {   
    #         "name": "REGULAMENTO DOS ESTÁGIOS CURRICULARES SUPERVISIONADOS",
    #         "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1608522&id_orgao_publicacao=0",
    #         "filepath": "ESTAGIO_UTFPR.pdf"
    #     },
    # ]
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

        
        return chromadb.Client()

    client_chroma = get_chroma_client(chroma_url)

    llm = ChatOllama(model="llama3.2", temperature=0.5, base_url=ollama_url)
    model_name = "Alibaba-NLP/gte-multilingual-base"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs={'trust_remote_code': True}
        
    )
    
    chromadb_path = "./data" # CONFIG YOUR PATH
    vector_store = Chroma(
        client=client_chroma,
        collection_name="rag",
        embedding_function=embeddings,
        persist_directory=chromadb_path
    )
    prep = PrepDocs(llm=llm, embedding=embeddings)
    prep.run(docs)
