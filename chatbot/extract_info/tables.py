import re
from difflib import SequenceMatcher
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd

SEP_ROW_RE = re.compile(r"^[\s|:\-]+$")

TABLE_MATCH_THRESHOLD = 0.6


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


def collectHtmlTables(soup) -> List[Dict]:
    """Tabelas do HTML bruto, com assinatura para casar com os elementos do unstructured.

    partition_html descarta rowspan/colspan, por isso as tabelas são lidas
    direto do HTML e depois reconciliadas por similaridade de texto.
    """
    tables = []

    for table in soup.find_all("table"):
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

    return tables
