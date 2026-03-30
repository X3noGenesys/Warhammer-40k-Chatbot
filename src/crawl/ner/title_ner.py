import json
import re
from collections import Counter
from pathlib import Path

INPUT_PATH  = "data/cleaned_data.jsonl"
OUTPUT_PATH = "data/title_entities.jsonl"

TYPE_RULES: list[tuple[str, list[str]]] = [
    ("EVENT", [
        r"\bguerre\b", r"\bbataille\b", r"\bheresie\b", r"\bherésie\b",
        r"\bsiege\b", r"\bsiège\b", r"\bcroisade\b", r"\bcampagne\b",
        r"\bassaut\b", r"\binvasion\b", r"\bchute\b", r"\bpurge\b",
        r"\brebellion\b", r"\binsurrection\b", r"\bconflit\b",
        r"\bmassacre\b", r"\bincursion\b", r"\braid\b",
    ]),
    ("LOCATION", [
        r"\bmonde\b", r"\bplanete\b", r"\bplanète\b", r"\bsysteme\b",
        r"\bsystème\b", r"\bsecteur\b", r"\bforteresse\b", r"\bcite\b",
        r"\bcité\b", r"\bruche\b", r"\bstation\b", r"\bnebuleuse\b",
        r"\bnébuleuse\b", r"\barchipel\b", r"\bregion\b", r"\brégion\b",
        r"\bterritoire\b", r"\bbastion\b", r"\bcitadelle\b",
    ]),
    ("ORGANIZATION", [
        r"\bchapitre\b", r"\blegion\b", r"\blégion\b", r"\bordre\b",
        r"\bregiment\b", r"\brégiment\b", r"\bcompagnie\b", r"\bcohort\b",
        r"\bbrigade\b", r"\bescadron\b", r"\bfraternity\b", r"\bcabal\b",
        r"\bsociete\b", r"\bsociété\b", r"\bgarde\b", r"\bclan\b",
        r"\bconfrerie\b", r"\bconfrérie\b", r"\bcercle\b",
        r"\badeptus\b", r"\badepta\b", r"\bofficio\b", r"\blegio\b",
        r"\bordo\b",
    ]),
    ("FACTION", [
        r"\bespace\b.*\bmarine\b", r"\bspace marine\b", r"\bork\b",
        r"\btyranide\b", r"\bnecron\b", r"\bnécron\b", r"\beldar\b",
        r"\baeldari\b", r"\bdrukhari\b", r"\bt.au\b", r"\btau\b",
        r"\bchaos\b", r"\bdemon\b", r"\bdémon\b",
    ]),
    ("CONCEPT", [
        r"\bwarp\b", r"\bpsyker\b", r"\bimmatérium\b", r"\bimmaterium\b",
        r"\bpouvoir\b", r"\bdoctrine\b", r"\bprotocole\b", r"\brite\b",
        r"\britual\b", r"\bmagie\b", r"\bcorruption\b", r"\bmutation\b",
        r"\btechnologie\b", r"\bartifice\b",
    ]),
    ("ARTIFACT", [
        r"\bepee\b", r"\bépée\b", r"\bhache\b", r"\bmarteau\b",
        r"\blance\b", r"\barmure\b", r"\bbouclier\b", r"\brelique\b",
        r"\bartefact\b", r"\bcanon\b", r"\bvaisseau\b", r"\bnavire\b",
        r"\btitane\b", r"\btitan\b",
    ]),
    ("CHARACTER", [
        r"\bprimarque\b", r"\bprimarch\b", r"\binquisiteur\b",
        r"\binquisitor\b", r"\bchapelain\b", r"\blieutenant\b",
        r"\bcapitaine\b", r"\bseigneur\b", r"\bcomte\b",
    ]),
]

IGNORE_PATTERNS = [
    r"^categorie\b", r"^category\b", r"^liste\b", r"^list\b",
    r"^portail\b", r"^portal\b", r"^aide\b", r"^help\b",
    r"^modele\b", r"^template\b", r"^discussion\b",
    r"^utilisateur\b", r"^user\b",
]


def infer_type(title: str) -> str:
    title_lower = title.lower()

    for entity_type, patterns in TYPE_RULES:
        for pattern in patterns:
            if re.search(pattern, title_lower):
                return entity_type

    return "UNKNOWN"


def should_ignore(title: str) -> bool:
    title_lower = title.lower().strip()
    for pattern in IGNORE_PATTERNS:
        if re.match(pattern, title_lower):
            return True

    if len(title.strip()) < 2 or len(title.strip()) > 200:
        return True
    return False


def normalize_title(title: str) -> str:
    return title.strip()


def extract_title_entities(input_path: str, output_path: str):
    print("=" * 55)
    print("TITLE NER — Extraction depuis les titres")
    print("=" * 55)

    if not Path(input_path).exists():
        print(f"[ERREUR] Fichier absent : {input_path}")
        return

    seen_titles: dict[str, dict] = {}
    total_docs  = 0
    ignored     = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            total_docs += 1

            title = doc.get("title", "").strip()
            url   = doc.get("url",   "")

            if not title or should_ignore(title):
                ignored += 1
                continue

            title_norm = normalize_title(title)
            title_key  = title_norm.lower()

            if title_key not in seen_titles:
                seen_titles[title_key] = {
                    "text":       title_norm,
                    "type":       infer_type(title_norm),
                    "source_url": url,
                }

    entities = list(seen_titles.values())

    type_counts = Counter(e["type"] for e in entities)

    print(f"\n[STATS]")
    print(f"  Documents lus     : {total_docs:,}")
    print(f"  Documents ignores : {ignored:,}")
    print(f"  Entites uniques   : {len(entities):,}")
    print(f"\n  Par type :")
    for t, n in type_counts.most_common():
        bar = "█" * (n * 25 // max(type_counts.values()))
        print(f"    {t:<15} {n:>5}  {bar}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ent in entities:
            f.write(json.dumps(ent, ensure_ascii=False) + "\n")

    print(f"\n[SAVE] {output_path} ({len(entities):,} entites)")

    unknowns = [e for e in entities if e["type"] == "UNKNOWN"][:15]
    if unknowns:
        print(f"\n[INFO] Exemples UNKNOWN (a verifier) :")
        for e in unknowns:
            print(f"  '{e['text']}'")


if __name__ == "__main__":
    extract_title_entities(INPUT_PATH, OUTPUT_PATH)
