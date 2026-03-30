import json
import re
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD


ENTITIES_PATH  = "data/quality_entities.jsonl"
RELATIONS_PATH = "data/quality_relations.jsonl"
OUTPUT_DIR     = Path("kg_artifacts")

WH40K  = Namespace("http://warhammer40k.org/ontology#")
WH40KD = Namespace("http://warhammer40k.org/data#")

TYPE_TO_CLASS: dict[str, URIRef] = {
    "CHARACTER":    WH40K.Character,
    "FACTION":      WH40K.Faction,
    "LOCATION":     WH40K.Location,
    "ARTIFACT":     WH40K.Artifact,
    "EVENT":        WH40K.Event,
    "ORGANIZATION": WH40K.Organization,
    "CONCEPT":      WH40K.Concept,
    "UNKNOWN":      WH40K.UnknownEntity,
}

RELATION_TO_PROP: dict[str, URIRef] = {
    "EST_MEMBRE_DE":      WH40K.memberOf,
    "DIRIGE":             WH40K.leads,
    "EST_ENNEMI_DE":      WH40K.enemyOf,
    "EST_ALLIE_DE":       WH40K.allyOf,
    "A_PARTICIPE_A":      WH40K.participatedIn,
    "S_EST_DEROULE_A":    WH40K.tookPlaceAt,
    "EST_SITUE_A":        WH40K.locatedIn,
    "MANIE":              WH40K.wields,
    "A_CREE":             WH40K.createdBy,
    "VENERE":             WH40K.worships,
    "CONTROLE":           WH40K.controls,
    "EST_SOUS_GROUPE_DE": WH40K.subgroupOf,
    "EST_ORIGINAIRE_DE":  WH40K.originatesFrom,
    "A_VAINCU":           WH40K.defeated,
    "A_TRAHI":            WH40K.betrayed,
}


def slugify(text: str) -> str:
    """Nettoie une chaîne pour en faire un fragment d'URI valide."""
    text = text.lower().strip()
    text = re.sub(r"[éèêë]", "e", text)
    text = re.sub(r"[àâä]", "a", text)
    text = re.sub(r"[ùûü]", "u", text)
    text = re.sub(r"[ôö]", "o", text)
    text = re.sub(r"[îï]", "i", text)
    text = re.sub(r"[ç]", "c", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def get_entity_id(text: str, etype: str) -> str:
    """Génère l'ID local (ex: character_abaddon_le_fleau)."""
    return f"{etype.lower()}_{slugify(text)}"

def entity_uri(text: str, etype: str) -> URIRef:
    """Génère l'URI complète pour RDF."""
    return WH40KD[get_entity_id(text, etype)]

def generate_entity_mapping(entities: list[dict]) -> dict[str, str]:
    """
    Crée un dictionnaire indexé par noms et alias pointant vers l'ID.
    Utile pour l'Entity Linking dans un chatbot.
    """
    mapping = {}
    for ent in entities:
        eid = get_entity_id(ent["text"], ent["type"])
        
        mapping[ent["text"].lower()] = eid
        
        for alias in ent.get("aliases", []):
            if alias:
                mapping[alias.lower()] = eid
    return mapping


def build_ontology() -> Graph:
    g = Graph()
    g.bind("wh40k", WH40K)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    
    g.add((WH40K.Entity, RDF.type, OWL.Class))
    
    classes = {
        WH40K.Character: ("Personnage", "Individu nommé"),
        WH40K.Faction: ("Faction", "Race ou armée"),
        WH40K.Location: ("Lieu", "Planète ou système"),
    }
    for cls, (label, comment) in classes.items():
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.subClassOf, WH40K.Entity))
        g.add((cls, RDFS.label, Literal(label, lang="fr")))

    g.add((WH40K.name, RDF.type, OWL.DatatypeProperty))
    g.add((WH40K.description, RDF.type, OWL.DatatypeProperty))
    
    return g

def populate_graph(entities: list[dict], relations: list[dict]) -> Graph:
    g = Graph()
    g.bind("wh40k", WH40K)
    g.bind("wh40kd", WH40KD)

    entity_index: dict[str, URIRef] = {}

    for ent in entities:
        text, etype = ent["text"], ent["type"]
        cls = TYPE_TO_CLASS.get(etype, WH40K.UnknownEntity)
        uri = entity_uri(text, etype)
        entity_index[text.lower()] = uri

        g.add((uri, RDF.type, cls))
        g.add((uri, WH40K.name, Literal(text, lang="fr")))
        if ent.get("description"):
            g.add((uri, WH40K.description, Literal(ent["description"], lang="fr")))

    for rel in relations:
        src_uri = _resolve_entity(rel["source"], entity_index)
        tgt_uri = _resolve_entity(rel["target"], entity_index)
        prop = RELATION_TO_PROP.get(rel["relation"])

        if src_uri and tgt_uri and prop:
            g.add((src_uri, prop, tgt_uri))

    return g

def _resolve_entity(name: str, index: dict[str, URIRef]) -> URIRef | None:
    key = name.lower()
    if key in index: return index[key]
    for idx_key, uri in index.items():
        if key in idx_key or idx_key in key: return uri
    return None


def run():
    print("--- DÉMARRAGE DU BUILD ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement des fichiers JSONL...")
    try:
        with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
            entities = [json.loads(l) for l in f if l.strip()]
        with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
            relations = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError as e:
        print(f"Erreur : Fichier manquant. {e}")
        return

    print("Génération du mapping entités -> IDs...")
    mapping = generate_entity_mapping(entities)
    mapping_path = OUTPUT_DIR / "entity_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Mapping sauvegardé : {mapping_path}")

    onto_graph = build_ontology()
    onto_graph.serialize(destination=str(OUTPUT_DIR / "ontology.ttl"), format="turtle")

    data_graph = populate_graph(entities, relations)
    data_graph.serialize(destination=str(OUTPUT_DIR / "graph.ttl"), format="turtle")
    data_graph.serialize(destination=str(OUTPUT_DIR / "graph.nt"), format="nt")

    print(f"TERMINÉ : {len(data_graph)} triplets générés.")

if __name__ == "__main__":
    run()