import re
import unicodedata
from typing import Dict

# Siglas comuns demais para valer a expansão.
SKIPPED_ACRONYMS = {"UTFPR"}

# Uma sigla definida no documento: "Unidade Concedente de Estágio (UCE)".
DEFINITION_RE = re.compile(r"\(\s*([A-Z][A-Za-z]*[A-Z])(s?)\s*\)")

WORD_RE = re.compile(r"[\wÀ-ÿ]+(?:-[\wÀ-ÿ]+)*")

CONNECTORS = {
    "a", "à", "ao", "aos", "as", "às", "com", "da", "das", "de", "do", "dos",
    "e", "em", "na", "nas", "no", "nos", "o", "os", "para", "pela", "pelas",
    "pelo", "pelos",
}

MAX_EXPANSION_WORDS = 10

# Nota "(redação dada pela ...)": o nome da resolução não é expandido.
PROTECTED_RE = re.compile(r"\(redação dada pel[ao] [^)]*\)", re.IGNORECASE)


def plain(text: str) -> str:
    """Minusculas e sem acento, para comparar as letras da sigla com a expansao."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )


def isSubsequence(letters: str, text: str) -> bool:
    remaining = iter(text)
    return all(letter in remaining for letter in letters)


def findExpansion(before: str, acronym: str):
    """Expansao da sigla no texto que vem logo antes do "(SIGLA)"."""
    clause = re.split(r"[,;:.()\n]", before)[-1]
    words = WORD_RE.findall(clause)[-MAX_EXPANSION_WORDS:]
    letters = plain(acronym)

    for start in range(len(words) - 1, -1, -1):
        first = words[start]
        if first.lower() in CONNECTORS or not first[0].isupper():
            continue
        if plain(first[0]) != letters[0]:
            continue

        expansion = " ".join(words[start:])
        content = "".join(
            plain(word) for word in words[start:] if word.lower() not in CONNECTORS
        )
        if isSubsequence(letters, content) and len(words) - start >= 2:
            return expansion

    return None


def singularHead(expansion: str) -> str:
    """Primeira palavra no singular: "Trabalhos de Conclusão de Curso" -> "Trabalho ..."."""
    first, separator, rest = expansion.partition(" ")
    if re.search(r"[oa]s$", first):
        first = first[:-1]
    return first + separator + rest


def extractAcronyms(text: str) -> Dict[str, str]:
    """Siglas que o proprio documento define, com a primeira expansao encontrada."""

    acronyms = {}

    for match in DEFINITION_RE.finditer(text):
        acronym = match.group(1)
        if acronym in SKIPPED_ACRONYMS:
            continue

        # Prefere a forma normal do corpo à do cabeçalho em caixa alta.
        current = acronyms.get(acronym)
        if current and not current.isupper():
            continue

        expansion = findExpansion(text[:match.start()], acronym)

        # "(TCC)" é a sigla no singular; "(DCNs)" fica no plural.
        if expansion and not match.group(2):
            expansion = singularHead(expansion)

        if expansion and (not current or not expansion.isupper()):
            acronyms[acronym] = expansion

    return acronyms


def expandAcronyms(text: str, acronyms: Dict[str, str]) -> str:
    """Escreve por extenso a primeira ocorrencia de cada sigla no texto."""
    if not text or not acronyms:
        return text

    for acronym in sorted(acronyms, key=len, reverse=True):
        expansion = acronyms[acronym]

        if plain(expansion) in plain(text):
            continue

        protected = [span.span() for span in PROTECTED_RE.finditer(text)]
        pattern = re.compile(rf"(?<![\w(]){re.escape(acronym)}(s?)\b")
        match = next(
            (
                found for found in pattern.finditer(text)
                if not any(start <= found.start() < end for start, end in protected)
            ),
            None,
        )
        if not match:
            continue

        text = (
            text[:match.start()]
            + f"{expansion} ({acronym}{match.group(1)})"
            + text[match.end():]
        )

    return text
