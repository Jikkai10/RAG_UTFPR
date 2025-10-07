# RAG para documentos da UTFPR 

Construção de um chatbot com LLM que utiliza RAG para obter informações dos documentos oficiais da UTFPR

## Local:
### Requerimentos:

    sudo apt-get install poppler-utils libmagic-dev tesseract-ocr tesseract-ocr-por
    pip install -r requirements.txt && pip install paddlepaddle --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/

### Ollama:

    curl -fsSL https://ollama.com/install.sh | sh

#### Modelo:

    ollama pull llama3.1

mais informações: https://ollama.com

## Docker:
    docker compose up --build
Instalar NVIDIA container toolkit para o modelo rodar na gpu: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

