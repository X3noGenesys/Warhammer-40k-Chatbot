import json
import time
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, RDF, OWL
from rdflib.plugins.sparql import prepareQuery


GRAPH_PATH     = "kg_artifacts/graph.ttl"
ALIGNMENT_PATH = "kg_artifacts/alignment.ttl"
OUTPUT_NT      = "kg_artifacts/expanded.nt"
OUTPUT_STATS   = "kg_artifacts/kb_stats.json"

WH40K  = Namespace("http://warhammer40k.org/ontology#")
WH40KD = Namespace("http://warhammer40k.org/data#")

RULES = [
    ("symétrie_enemyOf", """
        CONSTRUCT { ?b wh40k:enemyOf ?a }
        WHERE {
            ?a wh40k:enemyOf ?b .
            FILTER NOT EXISTS { ?b wh40k:enemyOf ?a }
        }
    """),

    ("symétrie_allyOf", """
        CONSTRUCT { ?b wh40k:allyOf ?a }
        WHERE {
            ?a wh40k:allyOf ?b .
            FILTER NOT EXISTS { ?b wh40k:allyOf ?a }
        }
    """),

    ("presence_via_event", """
        CONSTRUCT { ?x wh40k:wasAt ?location }
        WHERE {
            ?x         wh40k:participatedIn ?event .
            ?event     wh40k:tookPlaceAt    ?location .
            FILTER NOT EXISTS { ?x wh40k:wasAt ?location }
        }
    """),

    ("membership_via_subgroup", """
        CONSTRUCT { ?x wh40k:memberOf ?parent }
        WHERE {
            ?x wh40k:memberOf  ?subgroup .
            ?subgroup wh40k:subgroupOf ?parent .
            FILTER NOT EXISTS { ?x wh40k:memberOf ?parent }
        }
    """),

    ("leader_is_member", """
        CONSTRUCT { ?x wh40k:memberOf ?org }
        WHERE {
            ?x wh40k:leads ?org .
            FILTER NOT EXISTS { ?x wh40k:memberOf ?org }
        }
    """),

    ("enemy_via_ally", """
        CONSTRUCT { ?x wh40k:enemyOf ?z }
        WHERE {
            ?x wh40k:allyOf  ?y .
            ?y wh40k:enemyOf ?z .
            FILTER (?x != ?z)
            FILTER NOT EXISTS { ?x wh40k:enemyOf ?z }
        }
    """),
    ("org_location_via_leader", """
        CONSTRUCT { ?org wh40k:locatedIn ?location }
        WHERE {
            ?leader wh40k:leads   ?org .
            ?leader wh40k:wasAt   ?location .
            FILTER NOT EXISTS { ?org wh40k:locatedIn ?location }
        }
    """),
]

def apply_rules(g: Graph) -> tuple[Graph, dict]:
    initNs = {"wh40k": WH40K, "wh40kd": WH40KD}
    stats  = {}
    total_new = 0

    print(f"\n[EXPAND] Application de {len(RULES)} règles d'expansion...")
    print(f"  Triplets avant expansion : {len(g)}")

    for rule_name, sparql in RULES:
        t0           = time.time()
        new_triples  = g.query(sparql, initNs=initNs)
        count_before = len(g)

        for s, p, o in new_triples:
            g.add((s, p, o))

        added   = len(g) - count_before
        elapsed = time.time() - t0
        stats[rule_name] = {"added": added, "elapsed_ms": round(elapsed * 1000)}
        total_new += added

        print(f"  [{rule_name}] +{added} triplets ({elapsed*1000:.0f}ms)")

    print(f"\n  Triplets après expansion : {len(g)} (+{total_new} inférés)")
    return g, stats

def compute_stats(g: Graph, expansion_stats: dict) -> dict:
    from collections import Counter

    class_counts: Counter = Counter()
    for _, _, cls in g.triples((None, RDF.type, None)):
        class_counts[str(cls).split("#")[-1]] += 1

    prop_counts: Counter = Counter()
    for _, prop, _ in g:
        prop_name = str(prop).split("#")[-1]
        prop_counts[prop_name] += 1

    entities = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        entities.add(s)

    outgoing: Counter = Counter()
    for s, p, o in g:
        if s in entities:
            outgoing[s] += 1

    avg_degree = sum(outgoing.values()) / len(outgoing) if outgoing else 0
    max_degree_uri = max(outgoing, key=outgoing.get) if outgoing else None
    max_degree_val = outgoing[max_degree_uri] if max_degree_uri else 0

    most_connected = None
    if max_degree_uri:
        for _, _, name in g.triples((max_degree_uri, WH40K.name, None)):
            most_connected = str(name)
            break

    return {
        "total_triples":       len(g),
        "total_entities":      len(entities),
        "avg_degree":          round(avg_degree, 2),
        "most_connected":      most_connected,
        "most_connected_degree": max_degree_val,
        "classes":             dict(class_counts.most_common()),
        "top_properties":      dict(prop_counts.most_common(15)),
        "expansion_rules":     expansion_stats,
    }


def print_stats(stats: dict):
    print(f"\n{'='*55}")
    print(f"STATISTIQUES DU KNOWLEDGE GRAPH")
    print(f"{'='*55}")
    print(f"Triplets total      : {stats['total_triples']:,}")
    print(f"Entités total       : {stats['total_entities']:,}")
    print(f"Degré moyen         : {stats['avg_degree']}")
    print(f"Entité la + liée    : {stats['most_connected']} "
          f"({stats['most_connected_degree']} relations)")
    print(f"\nInstances par classe :")
    for cls, n in stats["classes"].items():
        if not cls.startswith(("type", "Class", "Property")):
            print(f"  {cls:<20} {n:>5}")
    print(f"\nTop propriétés :")
    for prop, n in list(stats["top_properties"].items())[:10]:
        if not prop.startswith(("type", "label", "comment")):
            print(f"  {prop:<25} {n:>5}")


def run():
    print("=" * 55)
    print("SPARQL EXPAND — Expansion + Statistiques")
    print("=" * 55)

    print(f"\n[1/3] Chargement du graphe : {GRAPH_PATH}")
    g = Graph()
    g.bind("wh40k",  WH40K)
    g.bind("wh40kd", WH40KD)
    g.parse(GRAPH_PATH, format="turtle")
    print(f"  {len(g)} triplets chargés.")

    align_path = Path(ALIGNMENT_PATH)
    if align_path.exists():
        print(f"\n[2/3] Fusion avec l'alignement : {ALIGNMENT_PATH}")
        g.parse(str(align_path), format="turtle")
        print(f"  {len(g)} triplets après fusion.")
    else:
        print(f"\n[2/3] Alignement absent, ignoré ({ALIGNMENT_PATH}).")

    print("\n[3/3] Expansion par règles SPARQL CONSTRUCT...")
    g, expansion_stats = apply_rules(g)

    output_nt = Path(OUTPUT_NT)
    output_nt.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_nt), format="nt")
    print(f"\n  Sauvegardé : {output_nt}")

    stats = compute_stats(g, expansion_stats)
    print_stats(stats)

    stats_path = Path(OUTPUT_STATS)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n  Sauvegardé : {stats_path}")

    print(f"\n{'='*55}")
    print(f"TERMINÉ — {len(g)} triplets dans expanded.nt")


if __name__ == "__main__":
    run()
