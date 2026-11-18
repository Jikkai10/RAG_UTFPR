import re
from typing import List, Optional, Tuple

from extract_info.glossary import DEFINITION_RE, findExpansion

# Alteração "onde se lê: ... leia-se: ...".
AMENDMENT_RE = re.compile(
    r"onde\s+se\s+l[êe]\s*:?\s*(?P<old>.+?)\s*"
    r"(?:\b[IVXLC]+\s*[-–]\s*)?leia-se\s*:?\s*(?P<new>.+?)\s*"
    r"(?=(?:\b[IVXLC]+\s*[-–]\s*)?onde\s+se\s+l[êe]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Alteração "DE: ... PARA: ..." que reescreve o artigo inteiro.
REPLACEMENT_RE = re.compile(
    r"^\s*DE\s*:\s*$(?P<old>.+?)^\s*PARA\s*:\s*$(?P<new>.+?)(?=^\s*DE\s*:\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)

# A assinatura do SEI vem logo depois da redação nova e não faz parte dela.
SIGNATURE_RE = re.compile(
    r"^\s*\(?\s*(?:documento\s+)?assinado\s+eletronicamente", re.IGNORECASE | re.MULTILINE
)

ACRONYM_TOKEN_RE = re.compile(r"[A-Z]{2,}s?")

EDGE_PUNCTUATION = " \n\t;.,“”\"'"


def parseAmendments(resolutionText: str) -> List[Tuple[str, str]]:
    text = resolutionText or ""
    signature = SIGNATURE_RE.search(text)
    if signature:
        text = text[:signature.start()]

    return [
        (match.group("old").strip(EDGE_PUNCTUATION), match.group("new").strip(EDGE_PUNCTUATION))
        for regex in (AMENDMENT_RE, REPLACEMENT_RE)
        for match in regex.finditer(text)
    ]


def collapseAcronyms(text: str) -> str:
    """"Conselho de Graduação ... (COGEP)" -> "COGEP"."""
    for match in reversed(list(DEFINITION_RE.finditer(text))):
        expansion = findExpansion(text[:match.start()], match.group(1))
        if not expansion:
            continue
        written = re.search(
            r"\s+".join(map(re.escape, expansion.split())) + r"\s*$", text[:match.start()]
        )
        if written:
            text = text[:written.start()] + match.group(1) + match.group(2) + text[match.end():]
    return text


def tolerantPattern(old: str) -> Optional[re.Pattern]:
    """Casa o trecho antigo comparando só as palavras."""
    parts = []
    for token in re.findall(r"\w+", collapseAcronyms(old)):
        part = re.escape(token)
        if ACRONYM_TOKEN_RE.fullmatch(token):
            # No regulamento a sigla pode estar por extenso: "... Profissional (COGEP)".
            part = rf"(?:[^()\n]{{1,150}}?\(\s*)?{part}(?:\s*\))?"
        parts.append(part)
    return re.compile(r"\W+".join(parts)) if parts else None


def applyAmendments(original: str, resolutionText: str, source: str) -> Optional[str]:
    """Texto vigente do artigo, com a redação nova no lugar da antiga."""
    text = original
    applied = False

    for old, new in parseAmendments(resolutionText):
        pattern = tolerantPattern(old)
        match = pattern.search(text) if pattern else None
        if not match:
            continue

        text = text[:match.start()] + f"{new} (redação dada pela {source})" + text[match.end():]
        applied = True

    return text if applied else None
