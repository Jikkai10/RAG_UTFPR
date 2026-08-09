"""Avaliação do RAG da UTFPR com ragas.

Uso:
    python evaluate.py make-data   # consulta a API e grava data.json
    python evaluate.py eval        # avalia o data.json já existente
    python evaluate.py all         # faz as duas coisas

A configuração toda vem do .env (veja .env.example).

Usa a API nova do ragas 0.4 (ragas.metrics.collections). Essas métricas não passam
pelo ragas.evaluate() antigo, então cada exemplo é pontuado direto pelo ascore().
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8080").rstrip("/")

QUESTIONS_FILE = BASE_DIR / "aval_utfpr_rag.json"
DATA_FILE = BASE_DIR / "data.json"
RESULTS_FILE = BASE_DIR / "results.json"

CONTEXT_SOURCE = os.getenv("EVAL_CONTEXT_SOURCE", "all")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://models.github.ai/inference")
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")

CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "2"))
MAX_RETRIES = 5
REQUEST_TIMEOUT = 600


def requireEnv(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} não definida no .env")
    return value


def judgeApiKey():

    value = os.getenv("JUDGE_API_KEY") or os.getenv("GITHUB_TOKEN_MODELS")

    if not value:
        raise RuntimeError("JUDGE_API_KEY não definida no .env")

    return value


def buildMetrics():

    llm = llm_factory(
        JUDGE_MODEL,
        client=AsyncOpenAI(
            api_key=judgeApiKey(),
            base_url=JUDGE_BASE_URL,
            timeout=REQUEST_TIMEOUT,
        ),
    )
    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        trust_remote_code=True,
    )

    return [
        (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        (AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response")),
        (ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
    ]


def formatContext(doc):

    metadata = ", ".join(
        f"{key}: {value}"
        for key, value in (doc.get("metadata") or {}).items()
        if key != "id" and value is not None
    )

    return f"Fonte: {metadata}\nConteudo: {doc.get('content', '')}"


def login(session):
    response = session.post(
        f"{BASE_URL}/login",
        json={"email": requireEnv("EVAL_EMAIL"), "password": requireEnv("EVAL_PASSWORD")},
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"login falhou ({response.status_code}): {response.text}")

    return response.json()["access_token"]


def makeData():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {login(session)}"

    data = []

    for position, item in enumerate(questions, start=1):
        print(f"[{position}/{len(questions)}] {item['question']}")

        response = session.post(
            f"{BASE_URL}/rag/eval/{uuid.uuid4()}",
            json={"message": item["question"]},
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"/rag/eval falhou na pergunta {position} "
                f"({response.status_code}): {response.text}"
            )

        result = response.json()

        data.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts_all": [formatContext(doc) for doc in result["contexts"]],
            "contexts_cited": [formatContext(doc) for doc in result["sources"]],
            "ground_truth": item["ground_truth"],
        })

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return data


def loadData():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def toRagasRows(data):

    key = "contexts_cited" if CONTEXT_SOURCE == "cited" else "contexts_all"

    return [
        {
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item.get(key) or item.get("contexts") or [],
            "reference": item["ground_truth"],
        }
        for item in data
    ]


def jsonNumber(value):

    return None if value is None or np.isnan(value) else value


async def scoreOne(metric, inputs, semaphore):

    if "retrieved_contexts" in inputs and not inputs["retrieved_contexts"]:
        print(f"  {metric.name}: sem contexto recuperado, NaN")
        return float("nan")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                result = await metric.ascore(**inputs)

            return float(result.value) if result.value is not None else float("nan")

        except Exception as error:
            if attempt == MAX_RETRIES:
                print(f"  {metric.name} falhou depois de {MAX_RETRIES} tentativas: {error}")
                return float("nan")

            wait = min(2 ** attempt, 60)
            print(f"  {metric.name} erro (tentativa {attempt}), esperando {wait}s: {error}")
            await asyncio.sleep(wait)


async def scoreRow(position, total, row, metrics, semaphore):
    scores = {}

    for metric, fields in metrics:
        scores[metric.name] = await scoreOne(
            metric, {field: row[field] for field in fields}, semaphore
        )

    print(f"[{position}/{total}] {row['user_input'][:60]} -> {scores}")

    return scores


async def scoreAll(rows, metrics):
    semaphore = asyncio.Semaphore(CONCURRENCY)

    return await asyncio.gather(*[
        scoreRow(position, len(rows), row, metrics, semaphore)
        for position, row in enumerate(rows, start=1)
    ])


def runEval():
    data = loadData()

    if not data:
        raise RuntimeError(f"{DATA_FILE.name} está vazio")

    rows = toRagasRows(data)
    metrics = buildMetrics()

    scores = asyncio.run(scoreAll(rows, metrics))

    finalResult = {
        metric.name: float(np.nanmean([score[metric.name] for score in scores]))
        for metric, _ in metrics
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_model": JUDGE_MODEL,
                "context_source": CONTEXT_SOURCE,
                "n_examples": len(rows),
                "media": {
                    name: jsonNumber(value) for name, value in finalResult.items()
                },
                "por_exemplo": [
                    {
                        "question": row["user_input"],
                        **{name: jsonNumber(value) for name, value in score.items()},
                    }
                    for row, score in zip(rows, scores)
                ],
            },
            f,
            ensure_ascii=False,
            indent=4,
            allow_nan=False,
        )

    print(f"\nMédia ({len(rows)} exemplos, contextos: {CONTEXT_SOURCE}):")
    for name, value in finalResult.items():
        print(f"  {name}: {value:.4f}")

    return finalResult


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "all"

    if command == "make-data":
        makeData()
    elif command == "eval":
        runEval()
    elif command == "all":
        makeData()
        runEval()
    else:
        print(__doc__)
        sys.exit(1)
