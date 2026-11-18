"""Avaliação do RAG da UTFPR com ragas.

Uso:
    python evaluate.py make-data   # consulta a API e grava data.json
    python evaluate.py eval        # avalia o data.json já existente
    python evaluate.py all         # faz as duas coisas
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

from config import EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL, QUERY_PREFIX

BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8080").rstrip("/")


QUESTIONS_FILE = BASE_DIR / os.getenv("EVAL_QUESTIONS_FILE", "../avaliacao/perguntas_respostas.json")
DATA_FILE = BASE_DIR / os.getenv("EVAL_DATA_FILE", "data.json")
RESULTS_FILE = BASE_DIR / os.getenv("EVAL_RESULTS_FILE", "results.json")

CONTEXT_SOURCE = os.getenv("EVAL_CONTEXT_SOURCE", "all")

SELECTED_METRICS = {
    part.strip()
    for part in os.getenv("EVAL_METRICS", "").replace(";", ",").split(",")
    if part.strip()
}

# Posições (1-based) das perguntas a pular.
SKIP = {
    int(part)
    for part in os.getenv("EVAL_SKIP", "").replace(";", ",").split(",")
    if part.strip().isdigit()
}

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://models.github.ai/inference")
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "2"))
MAX_RETRIES = 3

REQUEST_TIMEOUT = int(os.getenv("EVAL_REQUEST_TIMEOUT", "600"))

# O padrão do ragas é 1024, que estoura em modelos de raciocínio
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "8192"))

JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "")

# Mesmo modelo e URL de api.py: o descarregamento é por modelo carregado.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


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


metricsCache = None


def buildMetrics():

    # Import local: o make-data não precisa do ragas.
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = llm_factory(
        JUDGE_MODEL,
        client=AsyncOpenAI(
            api_key=judgeApiKey(),
            base_url=JUDGE_BASE_URL,
            timeout=REQUEST_TIMEOUT,
        ),
        max_tokens=JUDGE_MAX_TOKENS,
        **({"reasoning_effort": JUDGE_REASONING_EFFORT} if JUDGE_REASONING_EFFORT else {}),
    )
    # Prefixo de query do e5 nos dois lados (pergunta com pergunta).
    promptKwargs = (
        {"prompts": {"query": QUERY_PREFIX}, "default_prompt_name": "query"}
        if "e5" in EMBEDDING_MODEL.lower()
        else {}
    )

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        trust_remote_code=True,
        **promptKwargs,
    )

    metrics = [
        (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        (AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response")),
        (ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
        (
            AnswerCorrectness(llm=llm, embeddings=embeddings),
            ("user_input", "response", "reference"),
        ),
    ]

    if not SELECTED_METRICS:
        return metrics

    desconhecidas = SELECTED_METRICS - {metric.name for metric, _ in metrics}

    if desconhecidas:
        raise RuntimeError(f"EVAL_METRICS desconhecida(s): {', '.join(sorted(desconhecidas))}")

    return [(metric, fields) for metric, fields in metrics if metric.name in SELECTED_METRICS]


def getMetrics():
    """Reaproveita as métricas entre chamadas."""
    global metricsCache

    if metricsCache is None:
        metricsCache = buildMetrics()

    return metricsCache


def formatValue(key, value):

    # Sem a instituição o juiz reprova afirmações com "na UTFPR".
    return f"{value} - UTFPR" if key == "source_type" else value


def formatContext(doc):

    metadata = ", ".join(
        f"{key}: {formatValue(key, value)}"
        for key, value in (doc.get("metadata") or {}).items()
        if key != "id" and value is not None
    )

    return f"Fonte: {metadata}\nConteudo: {doc.get('content', '')}"


def buildDataRow(question, groundTruth, result):
    """Monta uma linha do data.json a partir da resposta do RAG."""
    row = {
        "question": question,
        "answer": result["answer"],
        "contexts_all": [formatContext(doc) for doc in result.get("contexts") or []],
        "contexts_cited": [formatContext(doc) for doc in result.get("sources") or []],
        "ground_truth": groundTruth,
    }

    if result.get("steps"):
        row["steps"] = result["steps"]

    return row


def login(session):
    response = session.post(
        f"{BASE_URL}/login",
        json={"email": requireEnv("EVAL_EMAIL"), "password": requireEnv("EVAL_PASSWORD")},
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"login falhou ({response.status_code}): {response.text}")

    return response.json()["access_token"]


def unloadOllama():
    """Descarrega o modelo para limpar o prompt cache do ollama."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "keep_alive": 0},
            timeout=60,
        )
    except requests.RequestException as error:
        print(f"  aviso: não consegui descarregar o ollama: {error}")


def saveData(data):
    temporary = DATA_FILE.with_suffix(".json.tmp")

    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    os.replace(temporary, DATA_FILE)


def makeData():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    data = loadData() if DATA_FILE.exists() else []
    answered = {row["question"] for row in data}

    if answered:
        print(f"retomando: {len(answered)} respostas já em {DATA_FILE.name}")

    session = requests.Session()

    falhas = []

    for position, item in enumerate(questions, start=1):
        if item["question"] in answered:
            print(f"[{position}/{len(questions)}] já respondida, pulando")
            continue

        if position in SKIP:
            print(f"[{position}/{len(questions)}] em EVAL_SKIP, pulando sem responder")
            falhas.append((position, "pulada por EVAL_SKIP"))
            continue

        print(f"[{position}/{len(questions)}] {item['question']}")

        # Uma falha não interrompe as demais perguntas.
        try:
            # O token vence em 60 min
            session.headers["Authorization"] = f"Bearer {login(session)}"

            response = session.post(
                f"{BASE_URL}/rag/eval/{uuid.uuid4()}",
                json={"message": item["question"]},
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"/rag/eval devolveu {response.status_code}: {response.text[:300]}"
                )

            data.append(
                buildDataRow(item["question"], item["ground_truth"], response.json())
            )

            saveData(data)

        except Exception as error:
            print(f"  pergunta {position} falhou, seguindo sem ela: "
                  f"{type(error).__name__}: {error}")
            falhas.append((position, f"{type(error).__name__}: {error}"))

        unloadOllama()

    saveData(data)

    if falhas:
        print()
        print(f"{len(falhas)} pergunta(s) sem resposta em {DATA_FILE.name}:")
        for position, motivo in falhas:
            print(f"  [{position}] {motivo}")

    return data


def loadData():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def toRagasRows(data, contextSource=None):

    contextSource = contextSource or CONTEXT_SOURCE

    key = "contexts_cited" if contextSource == "cited" else "contexts_all"

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


async def scoreData(data, contextSource=None):
    """Pontua uma lista de exemplos já respondidos e devolve as médias e o detalhe."""
    contextSource = contextSource or CONTEXT_SOURCE

    rows = toRagasRows(data, contextSource)
    metrics = getMetrics()

    scores = await scoreAll(rows, metrics)

    media = {
        metric.name: float(np.nanmean([score[metric.name] for score in scores]))
        for metric, _ in metrics
    }

    # A média ignora os NaN; a contagem mostra quantos ficaram de fora.
    falhas = {
        metric.name: sum(1 for score in scores if np.isnan(score[metric.name]))
        for metric, _ in metrics
    }

    return {
        "judge_model": JUDGE_MODEL,
        "context_source": contextSource,
        "n_examples": len(rows),
        "media": {name: jsonNumber(value) for name, value in media.items()},
        "falhas": falhas,
        "por_exemplo": [
            {
                "question": row["user_input"],
                **{name: jsonNumber(value) for name, value in score.items()},
            }
            for row, score in zip(rows, scores)
        ],
    }


def mergeResults(anterior, parcial):
    """Junta um resultado parcial de métricas ao results.json já gravado."""
    porPergunta = {item["question"]: dict(item) for item in anterior.get("por_exemplo", [])}

    for item in parcial["por_exemplo"]:
        porPergunta.setdefault(item["question"], {"question": item["question"]}).update(item)

    return {
        **anterior,
        **{
            chave: valor
            for chave, valor in parcial.items()
            if chave not in ("media", "falhas", "por_exemplo")
        },
        "media": {**anterior.get("media", {}), **parcial["media"]},
        "falhas": {**anterior.get("falhas", {}), **parcial["falhas"]},
        "por_exemplo": list(porPergunta.values()),
    }


def runEval():
    data = loadData()

    if not data:
        raise RuntimeError(f"{DATA_FILE.name} está vazio")

    result = asyncio.run(scoreData(data))

    if SELECTED_METRICS and RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            result = mergeResults(json.load(f), result)

        print(f"\n(métricas {', '.join(sorted(SELECTED_METRICS))} costuradas em {RESULTS_FILE.name})")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4, allow_nan=False)

    print(f"\nMédia ({result['n_examples']} exemplos, contextos: {result['context_source']}):")
    for name, value in result["media"].items():
        falhas = result["falhas"][name]
        aviso = f"  <- {falhas}/{result['n_examples']} falharam, fora da média" if falhas else ""
        print(f"  {name}: {'nan' if value is None else f'{value:.4f}'}{aviso}")

    return result["media"]


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
