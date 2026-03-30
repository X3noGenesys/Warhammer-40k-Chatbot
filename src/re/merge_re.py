import json
from collections import Counter
from pathlib import Path

RELATIONS_PATH = "data/relations.jsonl"
NER_PATH       = "data/merged_ner.jsonl"
OUTPUT_PATH    = "data/merged_relations.jsonl"

def load_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        print(f"  [WARN] Fichier absent, ignoré : {path}")
        return []
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Sauvegardé : {path} ({len(records)} lignes)")


def build_entity_index(ner_docs: list[dict]) -> set[str]:
    known: set[str] = set()
    for doc in ner_docs:
        if "text" in doc:
            known.add(doc["text"].lower().strip())

        for ent in doc.get("entities", []):
            if "text" in ent:
                known.add(ent["text"].lower().strip())
    return known


def merge_relations(
    re_docs: list[dict],
    known_entities: set[str],
) -> list[dict]:
    merged: dict[tuple, dict] = {}

    total_raw = 0
    for doc in re_docs:
        url = doc.get("url", "")
        for rel in doc.get("relations", []):
            total_raw += 1
            key = (
                rel["source"].lower(),
                rel["relation"],
                rel["target"].lower(),
            )

            if key not in merged:
                merged[key] = {
                    "source":      rel["source"],
                    "relation":    rel["relation"],
                    "target":      rel["target"],
                    "source_urls": [],
                    "frequency":   0,
                }

            entry = merged[key]
            entry["frequency"] += 1
            if url and url not in entry["source_urls"]:
                entry["source_urls"].append(url)

    print(f"  Relations brutes (cross-docs) : {total_raw}")
    print(f"  Relations uniques             : {len(merged)}")

    results = []
    for (src_lower, relation, tgt_lower), entry in merged.items():
        entry["confirmed_by_ner"] = (
            src_lower in known_entities and
            tgt_lower in known_entities
        )
        results.append(entry)

    results.sort(key=lambda r: r["frequency"], reverse=True)
    return results

def print_stats(relations: list[dict]):
    rel_counter:  Counter = Counter()
    confirmed     = sum(1 for r in relations if r["confirmed_by_ner"])
    multi_source  = sum(1 for r in relations if r["frequency"] > 1)

    for r in relations:
        rel_counter[r["relation"]] += 1

    print(f"\n  Relations totales         : {len(relations)}")
    print(f"  Confirmées par NER        : {confirmed} "
          f"({confirmed/len(relations)*100:.1f}%)" if relations else "")
    print(f"  Mentionnées 2+ fois       : {multi_source}")
    print(f"\n  Par type :")
    for t, n in rel_counter.most_common():
        print(f"    {t:<25} {n:>5}")

    print(f"\n  Top 10 relations les plus fréquentes :")
    for r in relations[:10]:
        print(f"    [{r['frequency']}x] {r['source']} "
              f"—{r['relation']}→ {r['target']}")


def run():
    print("=" * 55)
    print("MERGE RE — Déduplication des relations")
    print("=" * 55)

    print("\n[1/3] Chargement des relations...")
    re_docs = load_jsonl(RELATIONS_PATH)
    print(f"  {len(re_docs)} documents avec relations")

    print("\n[2/3] Chargement de l'index d'entités...")
    ner_docs = load_jsonl(NER_PATH)
    known_entities = build_entity_index(ner_docs)
    print(f"  {len(known_entities)} entités connues")

    print("\n[3/3] Fusion et déduplication...")
    merged_relations = merge_relations(re_docs, known_entities)
    print_stats(merged_relations)

    save_jsonl(merged_relations, OUTPUT_PATH)

    print(f"\n{'='*55}")
    print(f"TERMINÉ → {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
