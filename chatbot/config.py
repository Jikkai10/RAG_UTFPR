from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# O e5 é treinado com prefixos assimétricos: a pergunta entra como "query: " e o
# trecho indexado como "passage: ". Sem eles os dois vetores caem no mesmo espaço
# simétrico e a similaridade fica pior.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def buildEmbeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"prompt": PASSAGE_PREFIX},
        query_encode_kwargs={"prompt": QUERY_PREFIX},
    )