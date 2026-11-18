# RAG para documentos da UTFPR 

Construção de um chatbot com LLM que utiliza RAG para obter informações dos documentos oficiais da UTFPR

O armazenamento (documentos, chunks, embeddings e usuários) é feito no **Neo4j**, usando um índice vetorial de 768 dimensões (compatível com o modelo de embedding `intfloat/multilingual-e5-base`).

## Configuração:

Copie o `.env.example` para `.env` e ajuste os valores:

    cp .env.example .env

O `docker-compose` também lê esse `.env` (`NEO4J_PASSWORD` e `JWT_SECRET`).

## Docker (mais simples):

    docker compose up --build

Isso sobe, nessa ordem: `neo4j` → `db_init` (cria constraints, índice vetorial e usuários padrão) → `rag_api` na porta 8080. O `ollama` sobe em paralelo e baixa o `llama3.2`.

O `db_init` é idempotente, roda a cada `up` sem apagar nada.

Para rodar o modelo na GPU (requer o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

    docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build

## Instalação local:

Recomendado para desenvolvimento: só os serviços no Docker, a API no host (evita rebuildar a imagem, que é pesada, a cada alteração).

### 1. Serviços:

    docker compose up -d neo4j ollama

### 2. Dependências do sistema:

    sudo apt-get install poppler-utils libmagic-dev tesseract-ocr tesseract-ocr-por

### 3. Dependências Python:

    pip install -r requirements.txt
    pip install paddlepaddle --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/

### 4. Schema do banco:

    python -m db.create_db

Cria as constraints, o índice vetorial e os usuários padrão (`admin@email.com` / `user@email.com`, senha `senha`). É idempotente.

Para zerar o banco (**apaga todos os nós, constraints e índices**):

    python -m db.create_db --reset

### 5. API:

    uvicorn api:app --host 0.0.0.0 --port 8080

### Ollama sem Docker:

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2

Os modelos testados foram llama3.1 e llama3.2. Para testes é recomendado o llama3.2 por ser mais leve e rápido, embora tenha pior desempenho.

Mais informações: https://ollama.com

## Uso:

A base começa vazia: antes de usar o chat é preciso popular os documentos, via `POST /docs` (por URL) ou `POST /upload-pdf/` (upload de arquivo).

O arquivo `extract_info/extract.py` tem uma lista de documentos de exemplo e pode ser rodado direto para popular a base:

    python -m extract_info.extract

## API:

Todas as rotas, exceto `/register` e `/login`, exigem um token JWT no header `Authorization: Bearer <token>`. As rotas de escrita de documentos (`POST /docs`, `POST /upload-pdf/`, `DELETE /docs/{doc_id}`) exigem um usuário com role `admin`.

Autenticação:
- `POST /login`: recebe `email` e `password`, retorna `access_token`
- `POST /register`: cria um usuário
- `GET /me`: dados do usuário do token

Chat:
- `GET /create_chat`: cria uma sessão e retorna o `thread_id`
- `GET /chat`: lista as sessões do usuário
- `GET /chat/{thread_id}`: histórico da sessão
- `PUT /chat/{thread_id}/{new_title}`: renomeia a sessão
- `DELETE /chat/{thread_id}`: apaga a sessão
- `POST /rag/{session_id}`: envia a mensagem e retorna a resposta gerada
- `POST /rag/stream/{session_id}`: mesma coisa, com resposta em streaming
- `POST /rag/eval/{session_id}`: retorna a resposta junto com os contextos recuperados (usado pelo `evaluate.py`)

Documentos (rotas de escrita são restritas a `admin`):
- `POST /docs`: popula a base a partir de uma URL (`name`, `url`, `doc_type`, `pai_id`)
- `POST /upload-pdf/`: popula a base a partir de um PDF enviado
- `GET /all_docs`: lista os documentos
- `GET /download/{doc_id}`: baixa o arquivo original
- `DELETE /docs/{doc_id}`: remove o documento

## Avaliação:

    python evaluate.py

Usa o `ragas` com um LLM juiz configurado no `.env` (`JUDGE_*`). Ver comentários no `.env.example`.

## Melhorias futuras:
- Melhoria das interfaces
- Substituir os usuários padrão por um fluxo de cadastro real
