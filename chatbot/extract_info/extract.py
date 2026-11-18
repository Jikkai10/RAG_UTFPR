import os

from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf

from config import UPLOAD_DIR, buildEmbeddings
from db.connection import Neo4jConnection
from extract_info.calendario import calSections, extractLayout
from extract_info.formula import FormulaReader
from extract_info.hierarchy import CHUNK_OVERLAP, CHUNK_SIZE, HierarchySplitter
from extract_info.layout import stableDocId
from extract_info.tables import collectHtmlTables
from extract_info.util import dictToList, insertCalendar, insertStructure


class PrepDocs:
    def __init__(self, llm, embedding):
        self.llm = llm
        self.embedding = embedding
        self.db = Neo4jConnection()
        self.formula = FormulaReader()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.hierarchy = HierarchySplitter(
            splitter=self.splitter,
            describe=self.getDescription,
            readFormula=self.formula.read,
        )

    def chunksToEmbeddings(self, chunk):
        return self.embedding.embed_documents([chunk])[0]

    def getDescription(self, table):
        prompt = (
            """
                forneça uma descrição simples e precisa da tabela a seguir em até 1000 caracteres, não forneça mais nada, apenas a descrição
            """

        )
        response = self.llm.invoke([SystemMessage(prompt)] + [f"{table}"])

        return response.content

    def insertDocument(self, doc, docs, docType):
        docList = dictToList(docs, self.chunksToEmbeddings)

        if docType == 0:
            doc["doc_id"] = None
        insertStructure(self.db, docList, docType, doc["doc_id"])

    def getDocument(self, doc, docType = 0):
        """
            partition html naturalmente exclui informações adicionais nas tags, como rowspan e colspan,
            o que pode ser problemático para descobrir o formato das tabelas,
            por isso o tratamento das tabelas é feito separadamente.
        """
        filePath = UPLOAD_DIR / doc["filename"]

        if not filePath.exists():
            raise FileNotFoundError("Arquivo não encontrado")

        with open(filePath, "r", encoding="utf-8") as f:
            htmlContent = f.read()

        tables = collectHtmlTables(BeautifulSoup(htmlContent, 'lxml'))

        elements = partition_html(
            filename=str(filePath),
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,

        )

        docs = self.hierarchy.split(elements, doc["name"], doc["filename"],
                                    stableDocId(filePath), tables)

        self.insertDocument(doc, docs, docType)

    def getPdfDocument(self, doc, docType = 0):
        """
            inferencia de tabelas pode sair com erros
        """
        filePath = UPLOAD_DIR / doc["filename"]
        elements = partition_pdf(
            filename=filePath,
            strategy="hi_res",
            languages=["por"],
            extract_images_in_pdf=True,
            include_page_breaks=False,
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )

        docs = self.hierarchy.split(elements, doc["name"], doc["filename"],
                                    stableDocId(filePath))

        self.insertDocument(doc, docs, docType)

    def getCalendarDocument(self, doc):
        path = UPLOAD_DIR / doc["filename"]
        docId = stableDocId(path)

        sections = calSections(extractLayout(path))

        for section in sections:
            section["id"] = f"{docId}_p{section['pagina']}"
            section.update(section.pop("info"))

            for month in section["meses"]:
                month["id"] = (
                    f"{section['id']}_m{month['mes_num'] or 0}_{month['periodo'] or 'x'}"
                )

                for index, item in enumerate(month["itens"]):
                    item["id"] = f"{month['id']}_i{index}"
                    item["chunks"] = [{
                        "id": f"{item['id']}_c0",
                        "texto": item.pop("chunk_texto"),
                    }]

            section["chunks"] = [
                {"id": f"{section['id']}_c{index}", "texto": text}
                for index, text in enumerate(section.pop("textos_chunk"))
            ]

        chunks = [
            chunk
            for section in sections
            for chunk in section["chunks"] + [
                c
                for month in section["meses"]
                for item in month["itens"]
                for c in item["chunks"]
            ]
        ]
        for chunk, embedding in zip(chunks, self.embedding.embed_documents(
            [chunk["texto"] for chunk in chunks]
        )):
            chunk["embedding"] = embedding

        monthCount = sum(len(section["meses"]) for section in sections)
        itemCount = sum(
            len(month["itens"]) for section in sections for month in section["meses"]
        )
        print(
            f"Calendário {doc['name']}: {len(sections)} seções, {monthCount} meses, "
            f"{itemCount} itens, {len(chunks)} chunks."
        )
        for section in sections:
            missing = [field for field in ("campus", "categoria", "ano") if not section.get(field)]
            if missing:
                print(f"  seção da página {section['pagina']} sem {', '.join(missing)}.")

        insertCalendar(self.db, {
            "doc_id": docId,
            "titulo": doc["name"],
            "path": doc["filename"],
            "secoes": sections,
        })

        return sections


    def run(self, docs, mode=1):
        print("Running prep docs...")

        for i, doc in enumerate(docs):
            print(f"Processing {i+1}/{len(docs)}: {doc['name']}")
            if mode == 1:
                self.getDocument(doc, 0)
            else:
                self.getPdfDocument(doc, 0)




if __name__ == "__main__":
    docs =[
        {
            "name": "REGULAMENTO DA ORGANIZAÇÃO DIDÁTICO-PEDAGÓGICA",
            "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1033898&id_orgao_publicacao=0",
        },
        {
            "name": "REGULAMENTO DOS ESTÁGIOS CURRICULARES SUPERVISIONADOS",
            "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1608522&id_orgao_publicacao=0",
        },

        {
            "name": "REGULAMENTO DE TRABALHO DE CONCLUSÃO DE CURSO",
            "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=3171226&id_orgao_publicacao=0",
        },

    ]

    # docs2=[
    #     {
    #         "name": "REGULAMENTO DOS ESTÁGIOS CURRICULARES SUPERVISIONADOS",
    #         "url": "https://sei.utfpr.edu.br/sei/publicacoes/controlador_publicacoes.php?acao=publicacao_visualizar&id_documento=1608522&id_orgao_publicacao=0",
    #         "filepath": "ESTAGIO_UTFPR.pdf"
    #     },
    # ]
    ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")

    llm = ChatOllama(model="llama3.2", temperature=0.5, base_url=ollamaUrl)

    embeddings = buildEmbeddings()

    prep = PrepDocs(llm=llm, embedding=embeddings)
    prep.run(docs)
