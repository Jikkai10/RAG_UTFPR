from unstructured.partition.pdf import partition_pdf
from PIL import Image
from langchain_ollama import OllamaEmbeddings
import os
import uuid
import chromadb
from langchain.vectorstores import Chroma
from langchain_ollama import ChatOllama
import ollama
from paddleocr import FormulaRecognition
from langchain_core.documents import Document
from pathlib import Path

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

def get_formula(img):
    
    model = FormulaRecognition(model_name="PP-FormulaNet_plus-M")
    output = model.predict(input=img, batch_size=1)
    for res in output:
        latex = res['rec_formula']
    return latex
        

def get_description(img):
    model = ChatOllama(model="gemma3")
    
    
    response = ollama.chat(
        model='gemma3',
        messages=[{
            'role': 'user',
            'content': """
                    Faça uma análise da imagem e retorne uma breve descrição
                    Caso identifique como uma formula, retorne apenas "sim" e nada mais
            
            """,
            'images': [Path(img)],
        }]
    )
    print(response['message']['content'])
    
    return response['message']['content']

def get_document(file_path):
    # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables
        languages=["por"],
        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        #extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=1500,  
        overlap=500,# defaults to 500
        #combine_text_under_n_chars=500,       # defaults to 0
        new_after_n_chars=1000,

        # extract_images_in_pdf=True,          # deprecated
    )
    
    
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    #images_b64.append(el.metadata.image_base64)
                    img = el.metadata.image_path
                    #desc = get_description(img)
                    #if desc == "sim":
                    desc = get_formula(img)
                    el.text = desc
                    chunk.text = ''
                    for el in chunk_els:
                        chunk.text += el.text+'\n'
    tables = []
    texts = []

    for chunk in chunks:
        # if "Table" in str(type(chunk)):
        #     tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk.text)             
    #print(chunks[0].metadata.to_dict())
    
    return [texts, chunks]

def get_embedding(text):
    
    
    embedding = embeddings.embed_documents([text])
    
    return embedding[0]

def prepare_documents(documents, document_name):
    
    embeddings = []
    metadatas = []
    for i, doc in enumerate(documents):
        embeddings.append(get_embedding(doc.text))
        metadatas.append({"source": document_name, "partition" : i})
        

    return embeddings, metadatas

def create_ids(documents):
    
    return [str(uuid.uuid4()) for _ in documents]

def insert_data(texts):#documents, embeddings, metadatas, ids):
    
    # collection.add(
    #     embeddings=embeddings,
    #     documents=documents,
    #     metadatas=metadatas,
    #     ids=ids
    # )
    vector_store.add_documents(
        [
            Document(
                page_content=text.text,
                metadata={
                    "source": text.metadata.filename,
                    "page_number": text.metadata.page_number,
                    
                },
                id=str(uuid.uuid4())
            )
            for text in texts
        ]
    )
    print(f"Data successfully entered! {len(texts)} Chunks")
    
def run():
    print("Running prep docs...")
    data_path = './data' # CONFIG YOUR PATH

    documents = []
    embeddings = []
    metadatas = []

    documents_names = os.listdir(data_path)
    documents_names_size = len(documents_names)
    for i, document_name in enumerate(documents_names): 
        print(f"{i+1}/{documents_names_size}: {document_name}")

        document_texts, document_chunks = get_document(os.path.join(data_path, document_name))
        #document_embeddings, document_metadatas = prepare_documents(document_chunks, document_name)
        documents.extend(document_chunks)
        #embeddings.extend(document_embeddings)
        #metadatas.extend(document_metadatas)
    
    ids = create_ids(documents)
    insert_data(documents)
    
if __name__ == "__main__":
    run()
