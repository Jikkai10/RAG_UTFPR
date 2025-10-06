# RAG para documentos da UTFPR 

Construção de um chatbot com LLM que utiliza RAG para obter informações dos documentos oficiais da UTFPR

### Requerimentos:

    sudo apt-get install poppler-utils libmagic-dev tesseract-ocr tesseract-ocr-por
    pip install -r requirements.txt

### Ollama:

    curl -fsSL https://ollama.com/install.sh | sh

#### Modelo:

    ollama pull llama3.1
docker exec -it ollama ollama pull llama3.1

mais informações: https://ollama.com
