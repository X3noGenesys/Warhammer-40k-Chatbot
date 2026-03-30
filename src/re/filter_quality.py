import json
from collections import Counter, defaultdict

TYPE_THRESHOLDS = {
    "FACTION": (2, 2),
    "ORGANIZATION": (2, 2),
    "EVENT": (2, 2),
    "CHARACTER": (3, 2),
    "LOCATION": (3, 2),
    "ARTIFACT": (4, 3),
    "CONCEPT": (10, 5),
}

DEFAULT_THRESHOLD = (5, 4)


ENTITIES_FILE = 'data/canonical_entities.jsonl'
RELATIONS_FILE = 'data/canonical_relations.jsonl'
OUTPUT_ENTITIES_FILE = 'data/quality_entities.jsonl'
OUTPUT_RELATIONS_FILE = 'data/quality_relations.jsonl'


def load_jsonl(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Erreur : Le fichier {filepath} est introuvable.")
    return data

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def display_statistics(entities, relations, title="Statistiques"):
    print(f"\n--- 📊 {title} ---")
    print(f"Total Entités : {len(entities)}")
    print(f"Total Relations : {len(relations)}")
    
    ent_types = Counter([e.get('type', 'UNKNOWN') for e in entities])
    print("\nRépartition par Type :")
    for t, count in ent_types.most_common():
        print(f"  - {t}: {count}")

def clean_graph_multi_criteria(entities, relations):
    print("\n--- 🧹 Nettoyage en cours ---")
    
    entity_type_map = {e['text']: e.get('type', 'UNKNOWN') for e in entities}
    

    mention_counts = Counter()
    entity_sources = defaultdict(set)
    
    for rel in relations:
        s, t = rel['source'], rel['target']
        urls = rel.get('source_urls', [])
        
        mention_counts[s] += 1
        mention_counts[t] += 1
        for url in urls:
            entity_sources[s].add(url)
            entity_sources[t].add(url)

    valid_entity_names = set()
    
    for name, m_count in mention_counts.items():
        e_type = entity_type_map.get(name, "UNKNOWN")
        
        min_m, min_s = TYPE_THRESHOLDS.get(e_type, DEFAULT_THRESHOLD)
        
        s_count = len(entity_sources[name])
        if m_count >= min_m and s_count >= min_s:
            valid_entity_names.add(name)

    cleaned_entities = [e for e in entities if e['text'] in valid_entity_names]
    cleaned_relations = [
        r for r in relations 
        if r['source'] in valid_entity_names and r['target'] in valid_entity_names
    ]

    return cleaned_entities, cleaned_relations

def main():
    entities = load_jsonl(ENTITIES_FILE)
    relations = load_jsonl(RELATIONS_FILE)

    if not entities: return

    display_statistics(entities, relations, "Stats Initiales")

    cleaned_entities, cleaned_relations = clean_graph_multi_criteria(entities, relations)

    display_statistics(cleaned_entities, cleaned_relations, "Stats Finales")

    save_jsonl(cleaned_entities, OUTPUT_ENTITIES_FILE)
    save_jsonl(cleaned_relations, OUTPUT_RELATIONS_FILE)
    print(f"\n✅ Nettoyage terminé.")

if __name__ == "__main__":
    main()