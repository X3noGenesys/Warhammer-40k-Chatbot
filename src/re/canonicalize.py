import json
import time
from collections import Counter, defaultdict
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path


NER_PATH       = "data/merged_ner.jsonl"
RELATIONS_PATH = "data/merged_relations.jsonl"

OUTPUT_ENTITIES_PATH  = "data/canonical_entities.jsonl"
OUTPUT_RELATIONS_PATH = "data/canonical_relations.jsonl"

FUZZY_THRESHOLD = 0.85
FUZZY_MIN_LENGTH = 6

GLOBAL_CANONICAL_MAP = {}
GLOBAL_ENTITY_DATA = {}
GLOBAL_ALL_KEYS = ()


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


def build_canonical_map(ner_docs: list[dict]):
    global GLOBAL_CANONICAL_MAP, GLOBAL_ALL_KEYS, GLOBAL_ENTITY_DATA
    mapping = {}
    entity_data = {}
    
    for doc in ner_docs:
        canonical = doc.get("text", "").strip()
        if not canonical:
            continue
            
        mapping[canonical.lower()] = canonical
        
        entity_data[canonical] = {
            "text": canonical,
            "type": doc.get("type", "CONCEPT"),
            "description": doc.get("description", ""),
            "source_urls": doc.get("source_urls", []),
            "aliases": doc.get("aliases", [])
        }
        
        for alias in doc.get("aliases", []):
            alias_clean = alias.strip().lower()
            if alias_clean:
                mapping[alias_clean] = canonical
                
    GLOBAL_CANONICAL_MAP = mapping
    GLOBAL_ENTITY_DATA = entity_data
    GLOBAL_ALL_KEYS = tuple(mapping.keys())
    print(f"  [MAP] {len(mapping):,} synonymes et formes canoniques chargés.")


@lru_cache(maxsize=20000)
def resolve_name(name: str) -> str:
    if not name:
        return ""
        
    name_lower = name.lower().strip()
    
    if name_lower in GLOBAL_CANONICAL_MAP:
        return GLOBAL_CANONICAL_MAP[name_lower]
        
    if len(name_lower) >= FUZZY_MIN_LENGTH:
        matches = get_close_matches(name_lower, GLOBAL_ALL_KEYS, n=1, cutoff=FUZZY_THRESHOLD)
        if matches:
            return GLOBAL_CANONICAL_MAP[matches[0]]
            
    return name.strip()


def process_data(relations: list[dict]):
    merged_rels: dict[tuple, dict] = {}
    used_canonical_names = set()
    
    print(f"\n  Normalisation de {len(relations):,} relations...")
    
    for i, rel in enumerate(relations):
        src_canon = resolve_name(rel["source"])
        tgt_canon = resolve_name(rel["target"])
        
        used_canonical_names.add(src_canon)
        used_canonical_names.add(tgt_canon)
        
        key = (src_canon.lower(), rel["relation"], tgt_canon.lower())
        
        if key not in merged_rels:
            merged_rels[key] = {
                "source":      src_canon,
                "relation":    rel["relation"],
                "target":      tgt_canon,
                "source_urls": set(rel.get("source_urls", [])),
                "frequency":   0,
            }
            
        entry = merged_rels[key]
        entry["frequency"] += rel.get("frequency", 1)
        for url in rel.get("source_urls", []):
            entry["source_urls"].add(url)
            
        if (i+1) % 10000 == 0:
            print(f"    ... {i+1:,} relations traitées")

    final_relations = []
    for entry in merged_rels.values():
        entry["source_urls"] = sorted(list(entry["source_urls"]))
        final_relations.append(entry)
    final_relations.sort(key=lambda x: x["frequency"], reverse=True)

    final_entities = []
    for name in sorted(used_canonical_names):
        if name in GLOBAL_ENTITY_DATA:
            final_entities.append(GLOBAL_ENTITY_DATA[name])
        else:
            final_entities.append({
                "text": name,
                "type": "CONCEPT",
                "description": "Entité extraite non répertoriée dans le NER.",
                "source_urls": [],
                "aliases": []
            })

    return final_entities, final_relations


def run():
    print("=" * 55)
    print("CANONICALIZE — Résolution et normalisation (Version V2)")
    print("=" * 55)

    print("\n[1/4] Chargement des données...")
    ner_docs = load_jsonl(NER_PATH)
    relations = load_jsonl(RELATIONS_PATH)
    
    print("\n[2/4] Construction de la table de résolution...")
    build_canonical_map(ner_docs)
    
    print("\n[3/4] Traitement global...")
    canonical_entities, canonical_relations = process_data(relations)
    
    print("\n[4/4] Sauvegarde...")
    save_jsonl(canonical_entities, OUTPUT_ENTITIES_PATH)
    save_jsonl(canonical_relations, OUTPUT_RELATIONS_PATH)
    
    print(f"\n  Entités canonicalisées : {len(canonical_entities):,}")
    print(f"  Relations finales      : {len(canonical_relations):,}")
    
    print(f"\nTERMINÉ → {OUTPUT_ENTITIES_PATH}")
    print(f"        → {OUTPUT_RELATIONS_PATH}")


if __name__ == "__main__":
    run()
