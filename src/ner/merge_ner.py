import json
from collections import Counter, defaultdict
from pathlib import Path

TITLE_ENTITIES_PATH  = "data/title_entities.jsonl"
SYNONYMS_MANUAL_PATH = "data/synonyms.jsonl"
OUTPUT_PATH          = "data/merged_ner.jsonl"

def load_jsonl(path: str) -> list[dict]:
    if not Path(path).exists():
        print(f"  [WARN] Absent, ignore : {path}")
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
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Sauvegarde : {path} ({len(records):,} lignes)")


def _is_valid_synonym(syn: str) -> bool:
    if not syn or len(syn.strip()) < 3:
        return False
    if "(" in syn or ")" in syn:
        return False
    return True


def merge(
    title_entities: list[dict],
    synonyms_manual: list[dict],
) -> list[dict]:
    
    reverse_map: dict[str, str] = {}
    canonical_data: dict[str, set[str]] = {}

    print(f"\n[1/3] Indexation de {len(synonyms_manual)} groupes de synonymes...")
    for doc in synonyms_manual:
        for item in doc.get("synonyms", []):
            canon = item.get("canonical", "").strip()
            if not canon: continue
            
            canon_lower = canon.lower()
            if canon_lower not in canonical_data:
                canonical_data[canon] = set()
            
            reverse_map[canon_lower] = canon
            canonical_data[canon].add(canon)
            
            for syn in item.get("synonyms", []):
                syn = syn.strip()
                if _is_valid_synonym(syn):
                    reverse_map[syn.lower()] = canon
                    canonical_data[canon].add(syn)

    merged_index: dict[str, dict] = {}
    
    print(f"[2/3] Fusion avec {len(title_entities)} entités crawlées...")
    
    for ent in title_entities:
        original_name = ent["text"].strip()
        original_name_lower = original_name.lower()
        
        target_canon = reverse_map.get(original_name_lower)
        
        if target_canon:
            if target_canon not in merged_index:
                merged_index[target_canon] = {
                    "text": target_canon,
                    "type": ent.get("type", "CONCEPT"),
                    "source_url": ent.get("source_url", ""),
                    "aliases": set(canonical_data[target_canon])
                }
            else:
                if not merged_index[target_canon]["source_url"]:
                    merged_index[target_canon]["source_url"] = ent.get("source_url", "")
        else:
            if original_name not in merged_index:
                merged_index[original_name] = {
                    "text": original_name,
                    "type": ent.get("type", "CONCEPT"),
                    "source_url": ent.get("source_url", ""),
                    "aliases": set()
                }

    print(f"[3/3] Ajout des entités manuelles manquantes...")
    for canon, aliases in canonical_data.items():
        if canon not in merged_index:
            merged_index[canon] = {
                "text": canon,
                "type": "CONCEPT",
                "source_url": "",
                "aliases": set(aliases)
            }

    result = []
    for entry in merged_index.values():
        if entry["text"] in entry["aliases"]:
            entry["aliases"].remove(entry["text"])
            
        result.append({
            "text": entry["text"],
            "type": entry["type"],
            "source_url": entry["source_url"],
            "aliases": sorted(list(entry["aliases"]))
        })
        
    return result



def run():
    print("=" * 55)
    print("MERGE NER")
    print("=" * 55)

    title_entities = load_jsonl(TITLE_ENTITIES_PATH)
    synonyms_manual = load_jsonl(SYNONYMS_MANUAL_PATH)

    entities = merge(title_entities, synonyms_manual)
    
    save_jsonl(entities, OUTPUT_PATH)
    print(f"\nTERMINE -> {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
