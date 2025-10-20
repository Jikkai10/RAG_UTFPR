from typing import List
import uuid
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import numpy as np
from pydantic import BaseModel
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate, RunConfig
import json
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import requests

class MessageRequest(BaseModel):
    message: str
    chat_history: List[dict] = []
    session_id: str

base_url = "http://localhost:8080"

llm_lang = ChatOllama(model="llama3.2",verbose=False,timeout=600,num_ctx=8192,disable_streaming=False)

model_name = "Alibaba-NLP/gte-multilingual-base"
embeddings_lang = HuggingFaceEmbeddings(
    model_name=model_name, 
    model_kwargs={'trust_remote_code': True}
    
)

embeddings = LangchainEmbeddingsWrapper(embeddings_lang)
llm = LangchainLLMWrapper(llm_lang)




def make_data():
    # arquivo json com perguntas em question" e resposta esperada em "ground_truth"
    with open("aval_utfpr_rag.json", "r") as f:
        aval = json.load(f)

    data =  []
    for item in aval:
        message = MessageRequest(message=item["question"], chat_history=[],  session_id=str(uuid.uuid4()))
        result = requests.post(base_url+"/rag", json=message.model_dump())
        
        if(result.status_code != 200):
            return
        result = result.json()
        
        data.append({
            "question": item["question"],
            "answer": result[-1]["content"],
            "contexts": [doc["page_content"] for doc in result[result.__len__()-2]["artifact"]],
            "ground_truth": item["ground_truth"]
        })
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4) 
        
    return data

def load_data():
    with open("data.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_data()

dataset = Dataset.from_list(data)

#para evitar runtime errors
run_config = RunConfig(timeout=600, max_retries=20, max_workers=2)

def batch_evaluate(full_dataset, batch_size=5):
    results_list = []
    for i in range(0, len(full_dataset), batch_size):
        batch = full_dataset.select(range(i, min(i + batch_size, len(full_dataset))))
        print(f"Avaliando exemplos {i+1} a {i+len(batch)}...")
        batch_result = evaluate(
            dataset=batch,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
        )
        
        print(batch_result)
        results_list.append(batch_result.scores[0])
        
    final_result = {}
    for metric in results_list[0].keys():
        final_result[metric] = np.mean([r[metric] for r in results_list])
    return final_result

results = batch_evaluate(dataset, batch_size=5)

print(results)


    

