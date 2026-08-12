import re
from datetime import date
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

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


def extractLayout(pdfPath) -> List[List[dict]]:
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


def calMonthText(month: dict) -> str:
    header = month["mes"] or "Sem mês definido"

    if month["dias_letivos"] is not None:
        header = f"{header} (dias letivos: {month['dias_letivos']})"

    return "\n".join(
        [f"## {header}"] + [f"- {item['texto']}" for item in month["itens"]]
    )


def calMonths(items: List[dict], info: dict, schoolDays: dict = None) -> List[dict]:
    """Agrupa os itens em meses, a unidade devolvida na busca.

    O chunk continua sendo um evento isolado — o mês apenas reúne os eventos
    que o chunk localizou, do mesmo jeito que o artigo reúne seus chunks nos
    regulamentos. A chave é (mês, período) porque as páginas de tabela trazem
    linhas de períodos letivos diferentes lado a lado.
    """
    months = []
    index = {}

    for item in items:
        item["texto"] = f"{calWhen(item)} — {item['descricao']}"

        key = (item["mes_num"], item["periodo"])
        month = index.get(key)

        if month is None:
            month = {
                "mes": item["mes"],
                "mes_num": item["mes_num"],
                "ano": item["ano"],
                "periodo": item["periodo"],
                "dias_letivos": (schoolDays or {}).get(item["mes_num"]),
                "contexto": calContext({**info, "periodo": item["periodo"]}),
                "itens": [],
            }
            index[key] = month
            months.append(month)

        month["itens"].append(item)

    for month in months:
        month["corpo"] = calMonthText(month)
        month["texto"] = f"# {month['contexto']}\n{month['corpo']}"

        context = month["contexto"]
        if month["mes"]:
            context = f"{context} · {month['mes']}"

        for item in month["itens"]:
            item["chunk_texto"] = calItemChunk(item, context)

    return months


def calSections(blocksByPage: List[List[dict]]) -> List[dict]:
    sections = []

    for page, blocks in enumerate(blocksByPage):
        info, body = calHeader(blocks)
        if not body:
            continue

        body = calColumns(body)

        blocksByMonth = calEvents(body, info)
        context = calContext(info)

        if blocksByMonth:
            months = calMonths(
                [item for month in blocksByMonth for item in month["itens"]],
                info,
                {month["mes_num"]: month["dias_letivos"] for month in blocksByMonth},
            )

            summary = "; ".join(
                f"{month['mes']}: {month['dias_letivos']}"
                for month in blocksByMonth
                if month["dias_letivos"] is not None
            )
            chunks = [f"{context}\nDias letivos por mês: {summary}"] if summary else []

            sections.append({
                "pagina": page,
                "tipo": "eventos",
                "info": info,
                "texto": "\n".join(
                    [f"# {context}"] + [month["corpo"] for month in months]
                ),
                "meses": months,
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
            items.extend(calTableItems(table, info))

        sections.append({
            "pagina": page,
            "tipo": "tabela",
            "info": info,
            "texto": "\n".join(text),
            "meses": calMonths(items, info),
            "textos_chunk": chunks,
        })

    return sections
