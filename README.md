# RAG para documentos da UTFPR 

Construção de um chatbot com LLM que utiliza RAG para obter informações dos documentos oficiais da UTFPR

## Instalação local:
### Requerimentos:

    sudo apt-get install poppler-utils libmagic-dev tesseract-ocr tesseract-ocr-por
    pip install -r requirements.txt
    pip install paddlepaddle --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/

### Ollama:

    curl -fsSL https://ollama.com/install.sh | sh

#### Modelo:

    ollama pull llama3.1

mais informações: https://ollama.com

## Docker:
    docker compose up --build
Instalar NVIDIA container toolkit para o modelo rodar na gpu: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Uso:

Caso rode com docker já vai fornecer a api no localhost:8080

Localmente rode:

    uvicorn api:app --host 0.0.0.0 --port 8080

## API:

- /rag: tipo post, retorna a resposta gerada, recebe 3 entradas:
    - message: mensagem a ser enviada
    - chat_history: histórico de mensagens (atualmente o histórico já é feito internamente, esse campo já tem o preenchimento padrão como [ ])
    - session_id: número inteiro que representa a sessão
- /docs: tipo post, popula a base de embeddings com os documentos, recebe uma lista de dicionário com os seguintes campos:
    - name: nome do documento
    - url: url do documento
- /chat: uma interface de chat simples para usar o modelo
- /docs_ui: uma interface simples para enviar as informações dos documentos

## Melhorias futuras:
- Permitir o envio de documentos em pdf (particionamento já implementado)
- Integração com um banco de dados para controle de documentos e usuários
- Melhoria das interfaces

