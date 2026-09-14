"""
"Lexical Retrieval A" from pipeline.jpg: pull the exact article(s) a question
cites out of a regulation markdown file.

No embeddings, no model. Every question in questions.json already tells us
which regulation ("regulation_target": "gdpr") and which articles
("regulation_articles": ["art_35"]) it is about, so the right text is found
by looking at the headings - a plain string lookup, which is what "lexical"
means here.

Each regulation is a markdown file in data/Regulations named
<regulation_target>.md. Articles are level-2 headings, but the four files
spell them differently:

    ## Articolo 2            (aiact.md, gdpr.md)
    ## Art. 3.               (legge_132_2025.md)
    ## Art. 69-quater        (legge_633_1941.md)
    ## ALLEGATO I            (aiact.md, the annexes)

and questions.json uses a third spelling: art_2, art_69_quater, allegato_I.
_normalise() maps all of these to one canonical key ("art_2",
"art_69_quater", "allegato_i") so they can be compared.
"""
import re
from functools import lru_cache

from rag import config

# Matches a level-2 heading and captures its text ("Articolo 2", "Art. 3.", ...).
HEADING = re.compile(r"^## (.+?)\s*$", re.MULTILINE)

# "Articolo 2" / "Art. 69-quater" / "Art. 3."  ->  number + optional latin suffix
ARTICLE_HEADING = re.compile(r"^(?:Articolo|Art\.?)\s*(\d+)(?:[-\s]?([a-z]+))?\.?$", re.IGNORECASE)
# "ALLEGATO I" / "allegato_I"  ->  roman numeral
ANNEX_HEADING = re.compile(r"^allegato[_\s]+([ivxlc]+)$", re.IGNORECASE)


def _normalise(label: str) -> str | None:
    """Turn any spelling of an article reference into one canonical key.

        "Articolo 2"      -> "art_2"
        "Art. 3."         -> "art_3"
        "Art. 69-quater"  -> "art_69_quater"
        "art_69_quater"   -> "art_69_quater"
        "ALLEGATO I"      -> "allegato_i"

    Returns None for a heading that is not an article (e.g. a chapter title),
    so the parser can skip it.
    """
    label = label.strip()
    # questions.json spelling "art_69_quater" -> "art 69 quater" so one regex fits both
    label = label.replace("_", " ")
    label = re.sub(r"^art\s+(\d+)\s+([a-z]+)$", r"Art. \1-\2", label, flags=re.IGNORECASE)
    label = re.sub(r"^art\s+(\d+)$", r"Art. \1", label, flags=re.IGNORECASE)

    m = ARTICLE_HEADING.match(label)
    if m:
        number, suffix = m.group(1), m.group(2)
        return f"art_{number}" + (f"_{suffix.lower()}" if suffix else "")
    m = ANNEX_HEADING.match(label)
    if m:
        return f"allegato_{m.group(1).lower()}"
    return None


def regulation_path(target: str):
    return config.REGULATIONS_DIR / f"{target}.md"


def regulation_exists(target: str) -> bool:
    """The 'Requirement regulation in uploaded corpus?' diamond in the flowchart."""
    return regulation_path(target).is_file()


@lru_cache(maxsize=None)
def parse_regulation(target: str) -> dict[str, dict]:
    """Split one regulation file into {canonical_key: {"heading", "text"}}.

    The text of an article runs from its heading to the next level-2 heading.
    Cached (lru_cache) because the files are big (aiact.md is 660 KB) and every
    question would otherwise re-read and re-split them. Upload a new version of
    a regulation -> call parse_regulation.cache_clear().
    """
    content = regulation_path(target).read_text(encoding="utf-8")
    headings = list(HEADING.finditer(content))

    articles = {}
    for i, match in enumerate(headings):
        key = _normalise(match.group(1))
        if key is None:
            continue
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        articles[key] = {
            "heading": match.group(1).strip(),
            "text": content[start:end].strip(),
        }
    return articles


def get_articles(target: str, article_ids: list[str]) -> list[dict]:
    """Return [{"id", "heading", "text"}] for each requested article.

    Raises KeyError naming the missing id, so a typo in questions.json is a
    loud error instead of a silently empty context.
    """
    if not regulation_exists(target):
        raise FileNotFoundError(f"Regulation '{target}' not found in {config.REGULATIONS_DIR}")
    articles = parse_regulation(target)
    found = []
    for article_id in article_ids:
        key = _normalise(article_id)
        if key is None or key not in articles:
            raise KeyError(f"Article '{article_id}' not found in {target}.md")
        found.append({"id": article_id, "heading": articles[key]["heading"], "text": articles[key]["text"]})
    return found
