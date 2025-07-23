from unstructured.partition.html import partition_html
import uuid
from langchain_chroma import Chroma
from paddleocr import FormulaRecognition
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import base64
import numpy as np
import cv2 
import ollama
import re
from typing import List, Tuple, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "Alibaba-NLP/gte-multilingual-base"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, 
   
    model_kwargs={'trust_remote_code': True}
    
)

chromadb_path = "./db" # CONFIG YOUR PATH

vector_store = Chroma(
    #client=chroma_client,
    collection_name="rag",
    embedding_function=embeddings,
    persist_directory=chromadb_path
)


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
                    forneça uma descrição simples e precisa da tabela a seguir em até 500 caracteres, não forneça mais nada, apenas a descrição

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
    

def custom_split_by_hierarchy(full_text: str,
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
        result = maybe_split(article_text, current_cap, current_sec, art_number, nome, url)
        for text, meta in result:
            chunks.append(text)
            metadatas.append(meta)
        buff_art_lines = []
        art_number = None
    
    aux = "" 
    cont_table = 0  
    for el in full_text:
        
        if "Table" in str(type(el)):
            
            t = aux + tables[cont_table] + "\n"
            
            desc = get_description(t)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=0,  
                length_function=len,
            )
            texts = splitter.split_text(tables[cont_table])
            
            cont_table += 1
            for text in texts:
                text_tab = aux + "\n" + desc + "\n" + text.strip()
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

def is_table_empty(table):
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        for cell in cells:
            text = cell.get_text(strip=True)
            if text:  # se houver qualquer conteúdo não vazio
                return False
    return True 

def get_table_text(table):
    soup = BeautifulSoup(table, "html.parser")
    rows = soup.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows[0].find_all("td")]
    data_rows = rows[1:]

    # Armazena valores ativos de rowspan
    active_rowspans = {}

    frases = ""
    for i, row in enumerate(data_rows):
        cells = []
        col_index = 0
        tds = row.find_all(['td', 'th'])
        td_idx = 0
        
        while col_index < len(headers):
            # Verifica se temos rowspan de linha anterior para essa coluna
            if col_index in active_rowspans and active_rowspans[col_index]['rows_left'] > 0:
                cells.append(active_rowspans[col_index]['value'])
                active_rowspans[col_index]['rows_left'] -= 1
                col_index += 1
                continue
            
            if td_idx >= len(tds):
                return table
            cell = tds[td_idx]
            td_idx += 1
            
            
            value = cell.get_text(strip=True)
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            

            if rowspan > 1:
                active_rowspans[col_index] = {'value': value, 'rows_left': rowspan - 1}

            if colspan > 1:
                for _ in range(colspan):
                    cells.append(value)
                    col_index += 1
            else:
                cells.append(value)
                col_index += 1
            

        # Montar frase com as células correspondentes
        row_dict = dict(zip(headers, cells))
        frase = ""
        for header in headers:
            frase += f"{header}: {row_dict[header]}, "
        
        frases += frase + "\n"

    
    return frases

def get_document(doc):
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
        #url=doc["url"],
        text=resq.text,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        
    )

    return custom_split_by_hierarchy(elements, doc["name"], doc["url"], frases)



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

    # for i, doc in enumerate(documents):
    #     print(f"Chunk {i+1}/{len(documents)}: {doc}")
    insert_data(documents, metadatas)
    
if __name__ == "__main__":
    run()
