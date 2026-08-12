import re
from typing import Dict, List, Tuple

from unstructured.documents.elements import Image as ImageElement, Table as TableElement

from extract_info.layout import elementPosition, mergePositions
from extract_info.tables import getTableText, matchTable, tableRowsToChunks

CAP_RE   = re.compile(r"^\s*Cap[ií]tulo\s+([IVXLCDM]+)\b.*",  re.IGNORECASE)
SEC_RE   = re.compile(r"^\s*Se[cç][ãa]o\s+([IVXLCDM]+)\b.*", re.IGNORECASE)
ART_RE   = re.compile(r"^\s*Art(?:igo)?\.?\s*(\d+)[ºo]?\b.*", re.IGNORECASE)
REF_RE = re.compile(r"\bArt(?:igo)?\.?\s*(\d+)[ºo]?\b", re.IGNORECASE)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

IGNORED_CATEGORIES = {"Header", "Footer", "PageBreak", "PageNumber"}

MAX_CAPTION_LEN = 200


def organizeChunksByHierarchy(doc, info):

    chapterId = info["cap_id"]
    sectionId = info["sec_id"]
    contentId = info["cont_id"]


    # Capítulo
    chapter = doc["capitulos"].setdefault(chapterId, {
        "id": f"{doc['id']}_cap{chapterId}",
        "capitulo": chapterId,
        "secoes": {},
        "conteudos": {}
    })

    if sectionId != "—":
        section = chapter["secoes"].setdefault(sectionId, {
            "id": f"{chapter['id']}_sec{sectionId}",
            "secao": sectionId,
            "conteudos": {}
        })
    else:
        section = chapter


    article = section["conteudos"].setdefault(contentId, {
        "id": f"{section['id']}_cont{contentId}",
        "cont_num": contentId,
        "tipo": info["type"],
        "texto": info["text"],
        "refs": info["refs"],
        "pagina_inicio": info.get("pagina_inicio"),
        "pagina_fim": info.get("pagina_fim"),
        "bbox": info.get("bbox"),
        "chunks": []
    })

    # Chunk
    for i, chunk in enumerate(info["chunk"]):
        chunkId = f"{article['id']}_chunk{i}"
        article["chunks"].append({
            "id": chunkId,
            "texto": chunk
        })


class HierarchySplitter:
    """Quebra os elementos do documento na hierarquia capítulo → seção → artigo.

    Recebe as dependências de fora — o splitter de texto, a descrição das
    tabelas (LLM) e a leitura de fórmulas — para que o percurso dos elementos
    fique testável sem subir modelo nenhum.
    """

    def __init__(self, splitter, describe, readFormula):
        self.splitter = splitter
        self.describe = describe
        self.readFormula = readFormula

    def extractInfo(self, longText: str,
                    chapter: str,
                    section: str,
                    article: str,
                    name: str,
                    url: str,
                    positions: List[Tuple] = None) -> Dict:

        texts = self.splitter.split_text(longText)

        refs = []
        for ref in REF_RE.finditer(longText):
            number = ref.group(1)
            if number == article or number in refs:
                continue
            refs.append(number)

        startPage, endPage, bbox = mergePositions(positions or [])

        out = {
            "doc_id": name,
            "cap_id": chapter,
            "sec_id": section,
            "cont_id": article,
            "text": longText,
            "chunk": texts,
            "refs": refs,
            "path": url,
            "type": "artigo",
            "pagina_inicio": startPage,
            "pagina_fim": endPage,
            "bbox": bbox,
        }


        return out

    def tableChunks(self, table: str, caption: str, body: str, description: str) -> List[str]:
        chunks = [f"{caption}\n{description}".strip() if caption else description]
        chunks.extend(self.splitter.split_text(body))
        chunks.extend(tableRowsToChunks(table, caption))

        return list(dict.fromkeys(
            chunk for chunk in chunks if chunk and chunk.strip()
        ))

    def split(self, elements,
              name: str,
              url: str,
              docId: str,
              tables: List[Dict] = None,
              ) -> Dict:

        currentChapter = "—"   # travessão indica 'desconhecido' nos primeiros artigos
        currentSection = "—"

        docs = {
            "id": docId,
            "titulo": name,
            "path": url,
            "capitulos": {}
        }

        articleLineBuffer = []
        positionBuffer = []
        articleNumber = None

        def flushArticle():
            nonlocal articleLineBuffer, positionBuffer, articleNumber
            if articleNumber is None or not articleLineBuffer:
                return
            articleText = "\n".join(articleLineBuffer)
            info = self.extractInfo(articleText, currentChapter, currentSection, articleNumber,
                                    name, url, positionBuffer)
            organizeChunksByHierarchy(docs, info)
            articleLineBuffer = []
            positionBuffer = []
            articleNumber = None

        previousLine = ""
        tableCount = 0
        for el in elements:

            if isinstance(el, TableElement):

                table = matchTable(el, tables)
                if table is None:
                    html = getattr(el.metadata, "text_as_html", None)
                    table = getTableText(html) if html else None
                if table is None:
                    continue

                tableCount += 1

                caption = previousLine.strip()
                if len(caption) > MAX_CAPTION_LEN:
                    caption = ""
                previousLine = ""

                body = f"{caption}\n{table}".strip() if caption else table
                description = self.describe(body)

                chunks = self.tableChunks(table, caption, body, description)

                page, bbox = elementPosition(el)

                out = {
                    "doc_id": name,
                    "cap_id": currentChapter,
                    "sec_id": currentSection,
                    "cont_id": f"tab_{tableCount}",
                    "text": "\n".join(
                        part for part in (caption, description, table) if part
                    ),
                    "chunk": chunks,
                    "refs": [],
                    "path": url,
                    "type": "tabela",
                    "pagina_inicio": page,
                    "pagina_fim": page,
                    "bbox": bbox,
                }

                organizeChunksByHierarchy(docs, out)

                continue

            if isinstance(el, ImageElement):
                articleLineBuffer.append(self.readFormula(el.metadata.image_base64))
                positionBuffer.append(elementPosition(el))
                previousLine = ""
                continue

            if el.category in IGNORED_CATEGORIES:
                continue

            text = (el.text or "").strip()
            if not text:
                continue

            if (m := CAP_RE.match(text)):
                flushArticle()
                currentChapter = m.group(1)     # ex: 'I', 'II'…
                currentSection = "—"            # reinicia seção
                previousLine = ""
                continue

            if (m := SEC_RE.match(text)):
                flushArticle()
                currentSection = m.group(1)
                previousLine = ""
                continue

            if (m := ART_RE.match(text)):
                flushArticle()
                articleNumber = m.group(1)       # número do artigo

                articleLineBuffer.append(text)
                positionBuffer.append(elementPosition(el))
                previousLine = ""
                continue

            previousLine = text + "\n"

            if articleNumber:
                articleLineBuffer.append(text)
                positionBuffer.append(elementPosition(el))

        flushArticle()
        return docs
