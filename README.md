# RAG for UTFPR documents

## Requirements:

Python<3.12

### Paddleocr:

    pip install paddleocr==3.0.0

    python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

more information: https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html

### Unstructured:

    apt-get install poppler-utils tesseract-ocr libmagic-dev

    pip install -Uq "unstructured[all-docs]" pillow lxml pillow

more information: https://docs.unstructured.io/open-source/introduction/quick-start#mac-os-linux

### Ollama:

    curl -fsSL https://ollama.com/install.sh | sh

#### models:

    ollama pull llama3.1

    ollama pull nomic-embed-text

more information: https://ollama.com
