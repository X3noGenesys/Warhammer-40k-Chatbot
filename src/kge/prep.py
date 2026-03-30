import json
import random
import time
from collections import Counter
from pathlib import Path

from rdflib import Graph, Namespace, RDF, URIRef

GRAPH_PATH  = "kg_artifacts/expanded.nt"
OUTPUT_DIR  = Path("data/kge")

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10

SUBSET_SIZES = [20_000, 50_000, None]

RANDOM_SEED = 42

WH40K  = Namespace("http://warhammer40k.org/ontology#")
WH40KD = Namespace("http://warhammer40k.org/data#")


def uri_to_slug(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def load_triples(path: str) -> list[tuple[str, str, str]]:
    fmt = "nt" if path.endswith(".nt") else "turtle"
    print(f"[LOAD] Chargement de {path} ...")
    t0 = time.time()
    g = Graph()
    g.parse(path, format=fmt)
    print(f"  {len(g):,} triplets bruts chargés en {time.time()-t0:.1f}s")

    valid_props = {
        str(WH40K.memberOf),
        str(WH40K.leads),
        str(WH40K.enemyOf),
        str(WH40K.allyOf),
        str(WH40K.participatedIn),
        str(WH40K.tookPlaceAt),
        str(WH40K.locatedIn),
        str(WH40K.wields),
        str(WH40K.createdBy),
        str(WH40K.worships),
        str(WH40K.controls),
        str(WH40K.subgroupOf),
        str(WH40K.originatesFrom),
        str(WH40K.defeated),
        str(WH40K.betrayed),
        str(WH40K.wasAt),
        str(WH40K.potentialAllyOf),
    }

    wh40kd_prefix = str(WH40KD)
    triples = []
    skipped_literal = 0
    skipped_prop    = 0

    for s, p, o in g:
        p_str = str(p)

        if p_str not in valid_props:
            skipped_prop += 1
            continue

        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            skipped_literal += 1
            continue
        if not str(s).startswith(wh40kd_prefix) or not str(o).startswith(wh40kd_prefix):
            skipped_literal += 1
            continue

        triples.append((
            uri_to_slug(str(s)),
            uri_to_slug(str(p)),
            uri_to_slug(str(o)),
        ))

    print(f"  Ignorés (mauvaise prop)  : {skipped_prop:,}")
    print(f"  Ignorés (littéral/hors) : {skipped_literal:,}")
    print(f"  Triplets utilisables     : {len(triples):,}")
    return triples


def deduplicate(triples: list[tuple]) -> list[tuple]:
    unique = list(set(triples))
    print(f"  Après déduplication      : {len(unique):,} ({len(triples)-len(unique):,} doublons supprimés)")
    return unique


def compute_stats(triples: list[tuple]) -> dict:
    entities  = set()
    relations = Counter()

    for s, r, o in triples:
        entities.add(s)
        entities.add(o)
        relations[r] += 1

    return {
        "n_triples":   len(triples),
        "n_entities":  len(entities),
        "n_relations": len(relations),
        "top_relations": dict(relations.most_common(10)),
    }


def print_stats(stats: dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Triplets  : {stats['n_triples']:,}")
    print(f"{prefix}Entités   : {stats['n_entities']:,}")
    print(f"{prefix}Relations : {stats['n_relations']:,}")
    print(f"{prefix}Top relations :")
    for rel, n in stats["top_relations"].items():
        bar = "█" * min(30, n * 30 // max(stats["top_relations"].values()))
        print(f"  {rel:<25} {n:>6}  {bar}")


def split_triples(
    triples: list[tuple],
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    rng.shuffle(triples)

    by_rel: dict[str, list] = {}
    for t in triples:
        by_rel.setdefault(t[1], []).append(t)

    train, valid, test = [], [], []

    for rel, rel_triples in by_rel.items():
        n     = len(rel_triples)
        n_val = max(1, int(n * valid_ratio))
        n_tst = max(1, int(n * TEST_RATIO))

        if n < 3:
            train.extend(rel_triples)
            continue

        test.extend(rel_triples[:n_tst])
        valid.extend(rel_triples[n_tst:n_tst + n_val])
        train.extend(rel_triples[n_tst + n_val:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    return train, valid, test


def write_split(triples: list[tuple], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, r, o in triples:
            f.write(f"{s}\t{r}\t{o}\n")


def write_dataset(triples: list[tuple], out_dir: Path, label: str):
    """Crée train/valid/test dans out_dir."""
    train, valid, test = split_triples(triples)

    write_split(train, out_dir / "train.txt")
    write_split(valid, out_dir / "valid.txt")
    write_split(test,  out_dir / "test.txt")

    stats = compute_stats(triples)
    stats["splits"] = {
        "train": len(train),
        "valid": len(valid),
        "test":  len(test),
    }

    print(f"\n[{label}] → {out_dir}")
    print(f"  train : {len(train):,}")
    print(f"  valid : {len(valid):,}")
    print(f"  test  : {len(test):,}")
    print(f"  entités : {stats['n_entities']:,}  |  relations : {stats['n_relations']:,}")

    return stats


def run():
    print("=" * 55)
    print("KGE PREP — Préparation des datasets")
    print("=" * 55)

    t0 = time.time()

    raw     = load_triples(GRAPH_PATH)
    triples = deduplicate(raw)

    full_stats = compute_stats(triples)
    print_stats(full_stats, "FULL")

    if len(triples) < 1000:
        print("\n[WARN] Moins de 1000 triplets — vérifiez expanded.nt")

    all_stats = {}
    rng = random.Random(RANDOM_SEED)

    for size in SUBSET_SIZES:
        if size is None:
            label   = "full"
            subset  = triples
        else:
            label   = str(size // 1000) + "k"
            by_rel: dict[str, list] = {}
            for t in triples:
                by_rel.setdefault(t[1], []).append(t)

            subset = []
            target = min(size, len(triples))
            ratio  = target / len(triples)
            for rel_triples in by_rel.values():
                n_pick = max(1, int(len(rel_triples) * ratio))
                picked = rel_triples[:]
                rng.shuffle(picked)
                subset.extend(picked[:n_pick])
            rng.shuffle(subset)
            subset = subset[:target]

        out_dir = OUTPUT_DIR / label
        stats   = write_dataset(subset, out_dir, label.upper())
        all_stats[label] = stats

    stats_path = OUTPUT_DIR / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] Stats → {stats_path}")

    print(f"\n{'='*55}")
    print(f"TERMINÉ en {time.time()-t0:.1f}s")
    print(f"Datasets créés dans : {OUTPUT_DIR}/")
    for size_label in all_stats:
        s = all_stats[size_label]
        print(f"  {size_label:<6} : {s['n_triples']:>7,} triplets  "
              f"({s['splits']['train']:,} train / "
              f"{s['splits']['valid']:,} valid / "
              f"{s['splits']['test']:,} test)")


if __name__ == "__main__":
    run()
