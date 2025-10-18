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


CAP_RE   = re.compile(r"^\s*Cap[ií]tulo\s+([IVXLCDM]+)\b.*",  re.IGNORECASE)
SEC_RE   = re.compile(r"^\s*Se[cç][ãa]o\s+([IVXLCDM]+)\b.*", re.IGNORECASE)
ART_RE   = re.compile(r"^\s*Art(?:igo)?\.\s*(\d+)[ºo]?\b.*", re.IGNORECASE)

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
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.model = create_model("PP-FormulaNet_plus-M")




    def get_description(self,table):
        prompt = (
            """
                forneça uma descrição simples e precisa da tabela a seguir em até 1000 caracteres, não forneça mais nada, apenas a descrição
            """
            
        )
        response = self.llm.invoke([SystemMessage(prompt)] + [f"{table}"])
        
        return response.content


    def maybe_split(self, long_text: str,
                    capitulo: str,
                    secao: str,
                    artigo: str,
                    nome: str,
                    url: str) -> List[Tuple[str, Dict]]:
        """
        Se o artigo for muito grande, divide mantendo o cabeçalho.
        """
        if secao == "—":
            header = f"Capítulo {capitulo}\nArt. {artigo}\n\n"
        else:
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
        

    def custom_split_by_hierarchy(self,full_text: str,
                                nome: str,
                                url: str,
                                tables: List[str] = None,
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
            result = self.maybe_split(article_text, current_cap, current_sec, art_number, nome, url)
            for text, meta in result:
                chunks.append(text)
                metadatas.append(meta)
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
                # splitter = RecursiveCharacterTextSplitter(
                #     chunk_size=2000,
                #     chunk_overlap=0,  
                #     length_function=len,
                # )
                # texts = splitter.split_text(tables[cont_table])
                
                cont_table += 1
                #for text in texts:
                text_tab = aux + "\n" + desc + "\n" + table
                if current_sec == "—":
                    header = f"Capítulo {current_cap}\n\n"
                else:
                    header = f"Capítulo {current_cap}\nSeção {current_sec}\n\n"
                chunks.append(header + text_tab.strip())
                metadatas.append({"capitulo": current_cap, "secao": current_sec, "artigo": "-", "fonte": nome, "fonte_url": url})
                aux = ""
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
        return chunks, metadatas

    

    def get_document(self,doc):
        """
            partition html naturalmente exclui informações adicionais nas tags, como rowspan e colspan,
            o que pode ser problemático para descobrir o formato das tabelas,
            por isso o tratamento das tabelas é feito separadamente.
        """
        resq = requests.get(doc["url"])
        soup = BeautifulSoup(resq.content, 'lxml')
        tables = soup.find_all('table')
        tables = [str(table) for table in tables if not is_table_empty(table)] 
        frases = []
        for table in tables:
            frases.append(get_table_text(table))
            
        elements = partition_html(
            text=resq.text,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            
        )

        return self.custom_split_by_hierarchy(elements, doc["name"], doc["url"], frases)

    def get_pdf_document(self,doc):
        """
            inferencia de tabelas pode sair com erros
        """
        elements = partition_pdf(
            filename=doc["filepath"],
            strategy="hi_res",
            languages=["por"],
            extract_images_in_pdf=True,
            include_page_breaks=False,
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )

        return self.custom_split_by_hierarchy(elements, doc["name"], doc["url"])

    def insert_data(self,documents, metadatas):

        
        self.vector_store.add_documents(
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



    def run(self,docs, mode=1):
        print("Running prep docs...")

        documents = []
        metadatas = []

        for i, doc in enumerate(docs):
            print(f"Processing {i+1}/{len(docs)}: {doc['name']}")
            if mode == 1:
                chunks, meta = self.get_document(doc)
            else:
                chunks, meta = self.get_pdf_document(doc)

            documents.extend(chunks)
            metadatas.extend(meta)
        
        self.insert_data(documents, metadatas)
    
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
    prep = PrepDocs(vector_store=vector_store, llm=llm)
    prep.run(docs)
