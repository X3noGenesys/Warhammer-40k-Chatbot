import json
import time
import requests
import hashlib
from pathlib import Path
from collections import Counter

from rdflib import Graph, Namespace, URIRef, Literal, RDF, OWL, XSD, RDFS


GRAPH_PATH   = Path("kg_artifacts/graph.ttl")
OUTPUT_PATH  = Path("kg_artifacts/alignment.ttl")
CACHE_DIR    = Path(".cache/wikidata")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

WIKIDATA_DELAY  = 1.1
MIN_CONFIDENCE  = 0.75
MIN_FREQUENCY   = 3
MAX_ENTITIES    = 1500
MAX_RETRIES     = 3

HEADERS = {
    "User-Agent": "WH40K-KG-Aligner/3.0 (Research Bot; contact: samuel.carel@edu.devinci.fr)",
    "Accept": "application/json",
}


WH40K  = Namespace("http://warhammer40k.org/ontology#")
WH40KD = Namespace("http://warhammer40k.org/data#")
WD     = Namespace("http://www.wikidata.org/entity/")
OWL    = Namespace("http://www.w3.org/2002/07/owl#")


def get_cache_path(label: str, lang: str) -> Path:
    hash_key = hashlib.md5(f"{label}_{lang}".encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{hash_key}.json"

def load_from_cache(label: str, lang: str):
    path = get_cache_path(label, lang)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_to_cache(label: str, lang: str, data: list):
    path = get_cache_path(label, lang)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def query_wikidata(label: str, lang: str = "fr") -> list[dict]:
    cached = load_from_cache(label, lang)
    if cached is not None:
        return cached

    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?itemDescription WHERE {{
      ?item rdfs:label "{label}"@{lang} .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{lang},en" .
      }}
    }}
    LIMIT 3
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=10
            )
            if response.status_code == 429:
                time.sleep(WIKIDATA_DELAY * 10)
                continue
            response.raise_for_status()
            data = response.json()
            results = data.get("results", {}).get("bindings", [])
            
            candidates = []
            for r in results:
                candidates.append({
                    "qid": r["item"]["value"].split("/")[-1],
                    "label": r.get("itemLabel", {}).get("value", ""),
                    "description": r.get("itemDescription", {}).get("value", "")
                })
            save_to_cache(label, lang, candidates)
            return candidates
        except Exception:
            time.sleep(WIKIDATA_DELAY)
            
    return []


def compute_confidence(query_label: str, candidate: dict, entity_type: str) -> float:
    score = 0.0
    label_found = candidate["label"].lower()
    desc_found = candidate["description"].lower()
    query_lower = query_label.lower()

    if query_lower == label_found: score += 0.5
    elif query_lower in label_found: score += 0.3

    w40k_keywords = ["warhammer", "40,000", "40k", "imperium", "chaos", "primarch", "astartes", "fictif"]
    if any(kw in desc_found for kw in w40k_keywords):
        score += 0.4
    
    return min(score, 1.0)


def align_graph():
    print("=" * 60)
    print("ALIGN FILTERED — Wikidata Targeted Integration")
    print("=" * 60)

    source_g = Graph()
    source_g.parse(GRAPH_PATH, format="turtle")

    freq_counter = Counter()
    for s, p, o in source_g:
        if isinstance(s, URIRef) and str(s).startswith(str(WH40KD)):
            freq_counter[s] += 1
        if isinstance(o, URIRef) and str(o).startswith(str(WH40KD)):
            freq_counter[o] += 1

    entities_to_process = []
    for uri, freq in freq_counter.items():
        if freq < MIN_FREQUENCY:
            continue
            
        name = None
        for _, _, n in source_g.triples((uri, WH40K.name, None)):
            name = str(n)
            break
        
        etype = "CONCEPT"
        for _, _, t in source_g.triples((uri, RDF.type, None)):
            etype = str(t).split("#")[-1].upper()
            break
            
        if name:
            entities_to_process.append({
                "uri": uri,
                "name": name,
                "type": etype,
                "freq": freq
            })

    type_priority = {"CHARACTER": 0, "FACTION": 1, "LOCATION": 2, "EVENT": 3, "ARTIFACT": 4, "CONCEPT": 5}
    entities_to_process.sort(key=lambda x: (type_priority.get(x["type"], 99), -x["freq"]))

    final_list = entities_to_process[:MAX_ENTITIES]
    
    print(f"\n[1/2] Filtrage terminé :")
    print(f"  Entités totales trouvées : {len(freq_counter)}")
    print(f"  Entités filtrées (freq >= {MIN_FREQUENCY}) : {len(entities_to_process)}")
    print(f"  Entités sélectionnées pour alignement : {len(final_list)}")
    print(f"  Temps estimé : ~{round((len(final_list) * WIKIDATA_DELAY) / 60)} minutes")

    align_g = Graph()
    align_g.bind("wh40k", WH40K)
    align_g.bind("wd", WD)
    align_g.bind("owl", OWL)
    
    found_count = 0
    
    for i, ent in enumerate(final_list, 1):
        print(f"  Progress: {i}/{len(final_list)} (Found: {found_count})")

        cached_fr = load_from_cache(ent["name"], "fr")
        if cached_fr is not None:
            candidates = cached_fr
        else:
            candidates = query_wikidata(ent["name"], "fr")
            time.sleep(WIKIDATA_DELAY)

        if not candidates:
            cached_en = load_from_cache(ent["name"], "en")
            if cached_en is not None:
                candidates = cached_en
            else:
                candidates = query_wikidata(ent["name"], "en")
                time.sleep(WIKIDATA_DELAY)
        
        best_match = None
        best_score = 0
        for cand in candidates:
            score = compute_confidence(ent["name"], cand, ent["type"])
            if score > best_score:
                best_score = score
                best_match = cand
        
        if best_match and best_score >= MIN_CONFIDENCE:
            wd_uri = WD[best_match["qid"]]
            align_g.add((ent["uri"], OWL.sameAs, wd_uri))
            align_g.add((ent["uri"], WH40K.wikidataId, Literal(best_match["qid"])))
            align_g.add((ent["uri"], WH40K.alignmentScore, Literal(best_score, datatype=XSD.float)))
            found_count += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    align_g.serialize(destination=OUTPUT_PATH, format="turtle")
    
    print(f"\n[2/2] Alignement terminé ! {found_count} liens créés.")
    print(f"  Fichier généré : {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    align_graph()
