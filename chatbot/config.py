from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Prefixos assimétricos do e5.
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