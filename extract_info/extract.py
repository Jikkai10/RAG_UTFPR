from langchain_ollama import ChatOllama
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as ImageElement, Table as TableElement
import hashlib
from datetime import date
from difflib import SequenceMatcher
from langchain_core.messages import SystemMessage
from paddlex import create_model
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import base64
import numpy as np
import cv2
import re
from typing import List, Tuple, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pandas as pd
from io import StringIO
from extract_info.util import dictToList, insertStructure, insertCalendar
from config import UPLOAD_DIR
from db.connection import Neo4jConnection
import fitz  # PyMuPDF

CAP_RE   = re.compile(r"^\s*Cap[ií]tulo\s+([IVXLCDM]+)\b.*",  re.IGNORECASE)
SEC_RE   = re.compile(r"^\s*Se[cç][ãa]o\s+([IVXLCDM]+)\b.*", re.IGNORECASE)
ART_RE   = re.compile(r"^\s*Art(?:igo)?\.?\s*(\d+)[ºo]?\b.*", re.IGNORECASE)
REF_RE = re.compile(r"\bArt(?:igo)?\.?\s*(\d+)[ºo]?\b", re.IGNORECASE)

SEP_ROW_RE = re.compile(r"^[\s|:\-]+$")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

IGNORED_CATEGORIES = {"Header", "Footer", "PageBreak", "PageNumber"}

TABLE_MATCH_THRESHOLD = 0.6

MAX_CAPTION_LEN = 200


CALENDAR_MONTHS = {
    "JANEIRO": 1, "FEVEREIRO": 2, "MARÇO": 3, "MARCO": 3, "ABRIL": 4, "MAIO": 5,
    "JUNHO": 6, "JULHO": 7, "AGOSTO": 8, "SETEMBRO": 9, "OUTUBRO": 10,
    "NOVEMBRO": 11, "DEZEMBRO": 12,
}
MONTH_BY_NUMBER = {
    number: name for name, number in CALENDAR_MONTHS.items() if name != "MARCO"
}

CAL_YEAR_RE = re.compile(r"CALEND[ÁA]RIO\s+ACAD[ÊE]MICO\s+(\d{4})", re.IGNORECASE)
CAL_CAMPUS_RE = re.compile(r"^CAMPUS\s+(.+)$", re.IGNORECASE)
CAL_TERM_RE = re.compile(
    r"([1-4])\s*[ºo]?\s*PER[ÍI]ODO\s+LETIVO(?:\s*[–—-]\s*(\d{4}))?", re.IGNORECASE
)
CAL_SCHOOL_DAYS_RE = re.compile(r"Dias\s+letivos\s*:?\s*(\d+)", re.IGNORECASE)

CAL_MONTH_RE = re.compile(r"^(" + "|".join(CALENDAR_MONTHS) + r")\b", re.IGNORECASE)

CAL_DAY_RE = re.compile(
    r"^(\d{1,2}(?:\s*(?:a|e|,|até|–|—|-)\s*\d{1,2})*)(?:\s+(.*))?$", re.IGNORECASE
)

CAL_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$")

CAL_TERM_INPUT_PATTERNS = (
    re.compile(r"(?:^|/)\s*([1-4])\s*[ºo]?\s*$"),
    re.compile(r"([1-4])\s*[ºo]?\s*per[íi]odo", re.IGNORECASE),
)
CAL_YEAR_INPUT_RE = re.compile(r"(20\d{2})")

CAL_FOOTER_MAX_DIGITS = 3
CAL_FOOTER_MIN_HEIGHT_RATIO = 0.80

CAL_TITLE_MIN_LEN = 20
CAL_TITLE_MIN_UPPERCASE_RATIO = 0.9

CAL_LABEL_GAP = 30

CAL_SYNONYMS = (
    (re.compile(r"in[íi]cio do per[íi]odo letivo", re.IGNORECASE),
     "início das aulas, começo do semestre, volta às aulas, primeiro dia de aula"),
    (re.compile(r"t[ée]rmino do per[íi]odo letivo", re.IGNORECASE),
     "fim das aulas, término do semestre, último dia de aula, quando acabam as aulas"),
    (re.compile(r"recep[çc][ãa]o aos calouros", re.IGNORECASE),
     "recepção dos novos alunos, boas-vindas aos ingressantes"),
    (re.compile(r"resultados finais", re.IGNORECASE),
     "notas finais, divulgação das notas, fechamento das notas"),
    (re.compile(r"matr[íi]cula", re.IGNORECASE),
     "inscrição em disciplinas, rematrícula"),
    (re.compile(r"exame de sufici[êe]ncia", re.IGNORECASE),
     "prova de proficiência, aproveitamento de disciplina"),
)

CAL_COLUMN_GAP = 150


def isTableEmpty(table):
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        for cell in cells:
            text = cell.get_text(strip=True)
            if text:  # se houver qualquer conteúdo não vazio
                return False
    return True

def getTableText(table):
    try:
        dfs = pd.read_html(StringIO(table))
    except ValueError:
        return None

    if not dfs:
        return None

    df = dfs[0]

    if "<th" not in table.lower():
        df.columns = [
            "" if pd.isna(col) else str(col)
            for col in df.iloc[0]
        ]
        df = df[1:]

    if df.empty:
        return None

    return df.fillna("").to_markdown(index=False)

def stableDocId(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:32]

def tableSignature(text: str) -> str:
    return re.sub(r"[^0-9a-zà-ÿ]+", "", (text or "").lower())

def matchTable(element, tables: Optional[List[Dict]]) -> Optional[str]:
    if not tables:
        return None

    target = tableSignature(element.text)
    if not target:
        return None

    best = None
    bestScore = 0.0

    for candidate in tables:
        if candidate["used"]:
            continue

        matcher = SequenceMatcher(None, target, candidate["signature"])
        if matcher.real_quick_ratio() < TABLE_MATCH_THRESHOLD:
            continue
        if matcher.quick_ratio() < TABLE_MATCH_THRESHOLD:
            continue

        score = matcher.ratio()
        if score > bestScore:
            best = candidate
            bestScore = score

    if best is None or bestScore < TABLE_MATCH_THRESHOLD:
        return None

    best["used"] = True
    return best["markdown"]

def tableRowsToChunks(markdownTable: str, caption: str = "") -> List[str]:
    rows = []
    for line in markdownTable.splitlines():
        line = line.strip()
        if not line.startswith("|") or SEP_ROW_RE.match(line):
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])

    if len(rows) < 2:
        return []

    header, *body = rows
    prefix = f"{caption.strip()}\n" if caption.strip() else ""

    chunks = []
    for row in body:
        pairs = [
            f"{column}: {value}"
            for column, value in zip(header, row)
            if value
        ]
        if pairs:
            chunks.append(prefix + " | ".join(pairs))

    return chunks

def elementPosition(element) -> Tuple[Optional[int], Optional[List[float]]]:
    meta = getattr(element, "metadata", None)
    page = getattr(meta, "page_number", None)
    coords = getattr(meta, "coordinates", None)

    bbox = None
    if coords is not None and coords.points and coords.system is not None:
        xs = [point[0] for point in coords.points]
        ys = [point[1] for point in coords.points]
        width = coords.system.width or 1
        height = coords.system.height or 1
        bbox = [
            min(xs) / width,
            min(ys) / height,
            max(xs) / width,
            max(ys) / height,
        ]

    return page, bbox

def mergePositions(positions: List[Tuple]) -> Tuple[Optional[int], Optional[int], Optional[List[float]]]:
    pages = [page for page, _ in positions if page is not None]
    start = min(pages) if pages else None
    end = max(pages) if pages else None

    boxes = [
        bbox for page, bbox in positions
        if bbox and (start is None or page == start)
    ]

    bbox = None
    if boxes:
        bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    return start, end, bbox

def calClean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"_{3,}", " ", text or "")).strip()


def calLines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


def calDays(spec: str) -> Optional[Tuple[int, int]]:
    numbers = [int(number) for number in re.findall(r"\d{1,2}", spec or "")]
    if not numbers or any(number < 1 or number > 31 for number in numbers):
        return None
    return min(numbers), max(numbers)


def calDate(year: Optional[int], month: Optional[int], day: Optional[int]) -> Optional[str]:
    try:
        return date(year, month, day).isoformat()
    except (TypeError, ValueError):
        return None


def calMonth(line: str) -> Tuple[Optional[int], str]:
    line = calClean(line)
    match = CAL_MONTH_RE.match(line)

    if not match:
        return None, line

    return CALENDAR_MONTHS[match.group(1).upper()], line[match.end():].strip()


def calBlockMonth(block: dict) -> Optional[int]:
    for line in calLines(block.get("text")):
        monthNumber, _ = calMonth(line)
        if monthNumber:
            return monthNumber

    return None


def calIsFooter(block: dict) -> bool:
    text = calClean(block.get("text"))
    height = block.get("height") or 0

    return (
        text.isdigit()
        and len(text) <= CAL_FOOTER_MAX_DIGITS
        and height > 0
        and block.get("y0", 0) >= height * CAL_FOOTER_MIN_HEIGHT_RATIO
    )


def calTitle(block: dict) -> Optional[str]:
    if len(calLines(block.get("text"))) != 1:
        return None

    text = calClean(block.get("text"))
    letters = [character for character in text if character.isalpha()]

    if len(text) < CAL_TITLE_MIN_LEN or not letters:
        return None

    uppercaseRatio = sum(1 for character in letters if character.isupper()) / len(letters)

    return text if uppercaseRatio >= CAL_TITLE_MIN_UPPERCASE_RATIO else None


def calHeader(blocks: List[dict]) -> Tuple[dict, List[dict]]:
    info = {"campus": None, "categoria": None, "periodo": None, "ano": None}
    body = []

    for block in blocks:
        if calIsFooter(block):
            continue

        text = calClean(block.get("text"))
        if not text:
            continue

        if body or calBlockMonth(block):
            body.append(block)
            continue

        match = CAL_YEAR_RE.search(text)
        if match:
            info["ano"] = info["ano"] or int(match.group(1))
            continue

        match = CAL_CAMPUS_RE.match(text)
        if match:
            info["campus"] = calClean(match.group(1))
            continue

        match = CAL_TERM_RE.search(text)
        if match:
            info["periodo"] = match.group(1)
            info["ano"] = int(match.group(2)) if match.group(2) else info["ano"]
            continue

        if info["categoria"] is None:
            info["categoria"] = text
            continue

        body.append(block)

    return info, body


def calSynonyms(description: str) -> str:
    return "; ".join(
        text for pattern, text in CAL_SYNONYMS if pattern.search(description or "")
    )


def calItemChunk(item: dict, context: str) -> str:
    lines = [context, item["texto"]]
    synonyms = calSynonyms(item["descricao"])

    if synonyms:
        lines.append(f"({synonyms})")

    return "\n".join(lines)


def calColumns(body: List[dict]) -> List[dict]:
    columns = []

    for block in sorted(body, key=lambda block: block.get("x0", 0)):
        if columns and block.get("x0", 0) - columns[-1][-1].get("x0", 0) <= CAL_COLUMN_GAP:
            columns[-1].append(block)
        else:
            columns.append([block])

    return [
        block
        for column in columns
        for block in sorted(column, key=lambda block: block.get("y0", 0))
    ]


def calContext(info: dict) -> str:
    parts = [f"Calendário acadêmico {info['ano']}" if info.get("ano") else "Calendário acadêmico"]

    if info.get("campus"):
        parts.append(f"Campus {info['campus']}")
    if info.get("categoria"):
        parts.append(info["categoria"])
    if info.get("periodo"):
        parts.append(f"{info['periodo']}º período letivo")

    return " · ".join(parts)


def calWhen(item: dict) -> str:
    start, end = item.get("data_inicio"), item.get("data_fim")

    if start and end and start != end:
        return f"de {calDateBr(start)} a {calDateBr(end)}"
    if start:
        return calDateBr(start)

    month = item.get("mes")

    return f"dia {item.get('dia_texto')} de {month}" if month else f"dia {item.get('dia_texto')}"


def calDateBr(iso: str) -> str:
    year, month, day = iso.split("-")

    return f"{day}/{month}/{year}"


def calEvents(body: List[dict], info: dict) -> List[dict]:
    months = []
    currentMonth = None
    pending = None

    for block in body:
        for line in calLines(block.get("text")):
            line = calClean(line)

            monthNumber, rest = calMonth(line)
            if monthNumber:
                currentMonth = {
                    "mes": MONTH_BY_NUMBER[monthNumber],
                    "mes_num": monthNumber,
                    "dias_letivos": None,
                    "itens": [],
                }
                months.append(currentMonth)
                pending = None
                line = rest

            if not line or currentMonth is None:
                continue

            match = CAL_SCHOOL_DAYS_RE.search(line)
            if match:
                currentMonth["dias_letivos"] = int(match.group(1))
                continue

            match = CAL_DAY_RE.match(line)
            days = calDays(match.group(1)) if match else None

            if days:
                pending = {
                    "dia_texto": calClean(match.group(1)),
                    "descricao": calClean(match.group(2) or ""),
                    "mes": currentMonth["mes"],
                    "mes_num": currentMonth["mes_num"],
                    "ano": info.get("ano"),
                    "periodo": info.get("periodo"),
                    "data_inicio": calDate(info.get("ano"), currentMonth["mes_num"], days[0]),
                    "data_fim": calDate(info.get("ano"), currentMonth["mes_num"], days[1]),
                }
                currentMonth["itens"].append(pending)
                continue

            if pending is not None:
                pending["descricao"] = calClean(f"{pending['descricao']} {line}")
            elif currentMonth["itens"]:
                last = currentMonth["itens"][-1]
                last["descricao"] = calClean(f"{last['descricao']} {line}")

    for month in months:
        month["itens"] = [item for item in month["itens"] if item["descricao"]]

    return months


def calLabels(blocks: List[dict]) -> List[str]:
    groups = []

    for block in sorted(blocks, key=lambda block: block.get("x0", 0)):
        text = calClean(block.get("text"))
        if not text:
            continue

        if groups and block.get("x0", 0) - groups[-1]["x0"] <= CAL_LABEL_GAP:
            groups[-1]["texto"] = f"{groups[-1]['texto']} {text}"
        else:
            groups.append({"x0": block.get("x0", 0), "texto": text})

    return [group["texto"] for group in groups]


def calTables(body: List[dict]) -> List[dict]:
    tables = []
    current = None

    for block in sorted(body, key=lambda block: block.get("y0", 0)):
        cells = calLines(block.get("text"))
        if not cells:
            continue

        title = calTitle(block)
        if title:
            current = {"titulo": title, "rotulos": [], "linhas": []}
            tables.append(current)
            continue

        if current is None:
            current = {"titulo": None, "rotulos": [], "linhas": []}
            tables.append(current)

        if any(CAL_DATE_RE.match(cell) for cell in cells):
            current["linhas"].append(cells)
        elif current["linhas"]:
            current["linhas"].append(cells)
        else:
            current["rotulos"].append(block)

    for table in tables:
        table["rotulos"] = calLabels(table["rotulos"])

    return tables


def calTableItems(table: dict, info: dict) -> List[dict]:
    items = []
    labels = table.get("rotulos") or []

    for cells in table.get("linhas") or []:
        term = info.get("periodo")

        for index, cell in enumerate(cells):
            match = CAL_TERM_RE.search(cell) or re.match(r"^([1-4])\s*[ºo]$", cell)
            if match:
                term = match.group(1)

        for index, cell in enumerate(cells):
            match = CAL_DATE_RE.match(cell)
            if not match:
                continue

            day, month = int(match.group(1)), int(match.group(2))
            label = labels[index] if index < len(labels) else table.get("titulo")
            isoDate = calDate(info.get("ano"), month, day)

            items.append({
                "dia_texto": cell,
                "descricao": calClean(label or "Data do calendário"),
                "mes": MONTH_BY_NUMBER.get(month),
                "mes_num": month,
                "ano": info.get("ano"),
                "periodo": term,
                "data_inicio": isoDate,
                "data_fim": isoDate,
            })

    return items


def calTableText(table: dict) -> str:
    lines = [table["titulo"]] if table.get("titulo") else []
    labels = table.get("rotulos") or []

    for cells in table.get("linhas") or []:
        parts = [
            f"{labels[index]}: {cell}" if index < len(labels) else cell
            for index, cell in enumerate(cells)
        ]
        lines.append(" | ".join(parts))

    if not table.get("linhas") and labels:
        lines.extend(labels)

    return "\n".join(lines)


def calSections(blocksByPage: List[List[dict]]) -> List[dict]:
    sections = []

    for page, blocks in enumerate(blocksByPage):
        info, body = calHeader(blocks)
        if not body:
            continue

        body = calColumns(body)

        months = calEvents(body, info)
        context = calContext(info)

        if months:
            items = []
            text = [f"# {context}"]

            for month in months:
                header = month["mes"]
                if month["dias_letivos"] is not None:
                    header = f"{header} (dias letivos: {month['dias_letivos']})"
                text.append(f"## {header}")

                for item in month["itens"]:
                    item["texto"] = f"{calWhen(item)} — {item['descricao']}"
                    item["chunk_texto"] = calItemChunk(item, f"{context} · {month['mes']}")
                    text.append(f"- {item['texto']}")
                    items.append(item)

            summary = "; ".join(
                f"{month['mes']}: {month['dias_letivos']}"
                for month in months
                if month["dias_letivos"] is not None
            )
            chunks = [f"{context}\nDias letivos por mês: {summary}"] if summary else []

            sections.append({
                "pagina": page,
                "tipo": "eventos",
                "info": info,
                "texto": "\n".join(text),
                "itens": items,
                "textos_chunk": chunks,
            })
            continue

        tables = calTables(body)
        items = []
        text = [f"# {context}"]
        chunks = []

        for table in tables:
            tableBody = calTableText(table)
            if not tableBody:
                continue

            text.append(tableBody)
            chunks.append(f"{context}\n{tableBody}")

            for item in calTableItems(table, info):
                item["texto"] = f"{calWhen(item)} — {item['descricao']}"
                item["chunk_texto"] = calItemChunk(item, context)
                items.append(item)

        sections.append({
            "pagina": page,
            "tipo": "tabela",
            "info": info,
            "texto": "\n".join(text),
            "itens": items,
            "textos_chunk": chunks,
        })

    return sections


class PrepDocs:
    def __init__(self, llm, embedding):
        self.llm = llm
        self.model = create_model("PP-FormulaNet_plus-M")
        self.embedding = embedding
        self.db = Neo4jConnection()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def chunksToEmbeddings(self, chunk):
        return self.embedding.embed_query(chunk)

    def getDescription(self, table):
        prompt = (
            """
                forneça uma descrição simples e precisa da tabela a seguir em até 1000 caracteres, não forneça mais nada, apenas a descrição
            """

        )
        response = self.llm.invoke([SystemMessage(prompt)] + [f"{table}"])

        return response.content


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


    def organizeChunksByHierarchy(self, doc, info):

        docId = info["doc_id"]
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



    def customSplitByHierarchy(self, elements,
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
            self.organizeChunksByHierarchy(docs, info)
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
                description = self.getDescription(body)

                chunks = [f"{caption}\n{description}".strip() if caption else description]
                chunks.extend(self.splitter.split_text(body))
                chunks.extend(tableRowsToChunks(table, caption))
                chunks = list(dict.fromkeys(
                    chunk for chunk in chunks if chunk and chunk.strip()
                ))

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

                self.organizeChunksByHierarchy(docs, out)

                continue

            if isinstance(el, ImageElement):
                img = el.metadata.image_path

                imgBytes = base64.b64decode(el.metadata.image_base64)
                buf = np.frombuffer(imgBytes, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                output = self.model.predict(input=img, batch_size=1)
                text = ""
                for res in output:
                    text += res['rec_formula']
                text = "$$"+text+"$$ \n"

                articleLineBuffer.append(text)
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


        soup = BeautifulSoup(htmlContent, 'lxml')

        tables = []
        for table in soup.find_all('table'):
            if isTableEmpty(table):
                continue
            markdown = getTableText(str(table))
            if markdown is None:
                continue
            tables.append({
                "markdown": markdown,
                "signature": tableSignature(table.get_text(" ")),
                "used": False,
            })

        elements = partition_html(
            filename=str(filePath),
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,

        )

        docs = self.customSplitByHierarchy(elements, doc["name"], doc["filename"],
                                           stableDocId(filePath), tables)

        docList = dictToList(docs, self.chunksToEmbeddings)

        if docType == 0:
            doc["doc_id"] = None
        insertStructure(self.db, docList, docType, doc["doc_id"])

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

        docs = self.customSplitByHierarchy(elements, doc["name"], doc["filename"],
                                           stableDocId(filePath))
        docList = dictToList(docs, self.chunksToEmbeddings)

        if docType == 0:
            doc["doc_id"] = None
        insertStructure(self.db, docList, docType, doc["doc_id"])



    def extractLayout(self, pdfPath):
        doc = fitz.open(pdfPath)

        return [
            [
                {
                    "page": page.number,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "height": page.rect.height,
                    "text": text.strip(),
                }
                for x0, y0, x1, y1, text, _, _ in page.get_text("blocks")
            ]
            for page in doc
        ]


    def getCalendarDocument(self, doc):
        path = UPLOAD_DIR / doc["filename"]
        docId = stableDocId(path)

        sections = calSections(self.extractLayout(path))

        for section in sections:
            section["id"] = f"{docId}_p{section['pagina']}"
            section.update(section.pop("info"))

            for index, item in enumerate(section["itens"]):
                item["id"] = f"{section['id']}_i{index}"
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
            for chunk in section["chunks"] + [c for item in section["itens"] for c in item["chunks"]]
        ]
        for chunk, embedding in zip(chunks, self.embedding.embed_documents(
            [chunk["texto"] for chunk in chunks]
        )):
            chunk["embedding"] = embedding

        itemCount = sum(len(section["itens"]) for section in sections)
        print(
            f"Calendário {doc['name']}: {len(sections)} seções, {itemCount} itens, "
            f"{len(chunks)} chunks."
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

        documents = []
        metadatas = []

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
    modelName = "Alibaba-NLP/gte-multilingual-base"

    embeddings = HuggingFaceEmbeddings(
        model_name=modelName,
        model_kwargs={'trust_remote_code': True}

    )

    prep = PrepDocs(llm=llm, embedding=embeddings)
    prep.run(docs)
