import json
import time
from pathlib import Path

import numpy as np
import torch

from pykeen.pipeline import pipeline
from pykeen.triples  import TriplesFactory


DATA_DIR   = Path("data/kge")
OUTPUT_DIR = Path("models")

DATASET_SIZES = ["20k", "50k", "full"]

MODEL_CONFIGS = {
    "transe": {
        "model":            "TransE",
        "embedding_dim":    512,
        "model_kwargs": {
            "scoring_fct_norm": 1,
        },
        "optimizer":        "Adam",
        "optimizer_kwargs": {"lr": 0.001},
        "loss":             "MarginRankingLoss",
        "loss_kwargs":      {"margin": 1.0},
        "training_loop":    "SLCWA",
        "negative_sampler": "basic",
        "num_epochs":       200,
        "batch_size":       512,
    },
    "rotate": {
        "model":            "RotatE",
        "embedding_dim":    512,
        "model_kwargs":     {},
        "optimizer":        "Adam",
        "optimizer_kwargs": {"lr": 0.001},
        "loss":             "NSSALoss",
        "loss_kwargs":      {"margin": 9.0, "adversarial_temperature": 2.0},
        "training_loop":    "SLCWA",
        "negative_sampler": "bernoulli",
        "negative_sampler_kwargs": {"num_negs_per_pos": 10},
        "num_epochs":       200,
        "batch_size":       512,
    },
}

RANDOM_SEED = 42


def load_factories(size: str) -> tuple:
    base = DATA_DIR / size
    print(f"\n[LOAD] Dataset '{size}' depuis {base}")

    train_tf = TriplesFactory.from_path(
        path=str(base / "train.txt"),
        create_inverse_triples=False,
    )

    valid_tf = TriplesFactory.from_path(
        path=str(base / "valid.txt"),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
        create_inverse_triples=False,
    )

    test_tf = TriplesFactory.from_path(
        path=str(base / "test.txt"),
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
        create_inverse_triples=False,
    )

    print(f"  Entités   : {train_tf.num_entities:,}")
    print(f"  Relations : {train_tf.num_relations:,}")
    print(f"  Train     : {train_tf.num_triples:,}")
    print(f"  Valid     : {valid_tf.num_triples:,}")
    print(f"  Test      : {test_tf.num_triples:,}")

    return train_tf, valid_tf, test_tf


def train_model(
    model_name: str,
    size: str,
    train_tf: TriplesFactory,
    valid_tf: TriplesFactory,
    test_tf:  TriplesFactory,
) -> dict:
    cfg      = MODEL_CONFIGS[model_name]
    run_name = f"{model_name}_{size}"
    out_dir  = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"ENTRAÎNEMENT : {run_name.upper()}")
    print(f"  Modèle      : {cfg['model']}")
    print(f"  Dim         : {cfg['embedding_dim']}")
    print(f"  Epochs      : {cfg['num_epochs']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*55}")

    t0 = time.time()

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,

        model=cfg["model"],
        model_kwargs={
            "embedding_dim": cfg["embedding_dim"],
            **cfg.get("model_kwargs", {}),
        },

        optimizer=cfg["optimizer"],
        optimizer_kwargs=cfg["optimizer_kwargs"],

        loss=cfg["loss"],
        loss_kwargs=cfg.get("loss_kwargs", {}),

        training_loop=cfg["training_loop"],
        negative_sampler=cfg.get("negative_sampler", "basic"),
        negative_sampler_kwargs=cfg.get("negative_sampler_kwargs", {}),

        training_kwargs={
            "num_epochs":  cfg["num_epochs"],
            "batch_size":  cfg["batch_size"],
        },

        evaluation_kwargs={"batch_size": 256},

        random_seed=RANDOM_SEED,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    elapsed = time.time() - t0

    metrics  = result.metric_results.to_dict()
    realistic = metrics.get("both", {}).get("realistic", {})
    mrr      = realistic.get("inverse_harmonic_mean_rank", 0.0)
    hits1    = realistic.get("hits_at_1",  0.0)
    hits3    = realistic.get("hits_at_3",  0.0)
    hits10   = realistic.get("hits_at_10", 0.0)

    print(f"\n[RÉSULTATS] {run_name}")
    print(f"  MRR       : {mrr:.4f}")
    print(f"  Hits@1    : {hits1:.4f}")
    print(f"  Hits@3    : {hits3:.4f}")
    print(f"  Hits@10   : {hits10:.4f}")
    print(f"  Durée     : {elapsed:.1f}s")

    result.save_to_directory(str(out_dir))
    print(f"  Pipeline sauvegardé → {out_dir}")

    entity_emb   = result.model.entity_representations[0](
        indices=None
    ).detach().cpu().numpy()

    relation_emb = result.model.relation_representations[0](
        indices=None
    ).detach().cpu().numpy()

    np.save(str(out_dir / "embeddings_entity.npy"),   entity_emb)
    np.save(str(out_dir / "embeddings_relation.npy"), relation_emb)

    id_to_entity = {v: k for k, v in train_tf.entity_to_id.items()}
    id_to_rel    = {v: k for k, v in train_tf.relation_to_id.items()}
    (out_dir / "entity_labels.json").write_text(
        json.dumps(id_to_entity, ensure_ascii=False, indent=2)
    )
    (out_dir / "relation_labels.json").write_text(
        json.dumps(id_to_rel, ensure_ascii=False, indent=2)
    )

    print(f"  Embeddings ({entity_emb.shape}) → {out_dir}/embeddings_entity.npy")

    return {
        "run":       run_name,
        "model":     cfg["model"],
        "size":      size,
        "MRR":       round(mrr,   4),
        "Hits@1":    round(hits1,  4),
        "Hits@3":    round(hits3,  4),
        "Hits@10":   round(hits10, 4),
        "elapsed_s": round(elapsed, 1),
        "n_entities": train_tf.num_entities,
        "n_relations": train_tf.num_relations,
    }


def print_comparison_table(all_results: list[dict]):
    print(f"\n{'='*70}")
    print("TABLEAU COMPARATIF — KGE")
    print(f"{'='*70}")
    header = f"{'Modèle':<20} {'Size':<8} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8} {'s':>8}"
    print(header)
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<20} {r['size']:<8} "
              f"{r['MRR']:>8.4f} {r['Hits@1']:>8.4f} "
              f"{r['Hits@3']:>8.4f} {r['Hits@10']:>8.4f} "
              f"{r['elapsed_s']:>8.1f}")


def run():
    print("=" * 55)
    print("KGE TRAIN — TransE + RotatE × 3 tailles")
    print(f"Device : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    pipeline_start = time.time()

    for size in DATASET_SIZES:
        train_tf, valid_tf, test_tf = load_factories(size)

        for model_name in MODEL_CONFIGS:
            try:
                result = train_model(
                    model_name, size,
                    train_tf, valid_tf, test_tf
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n[ERREUR] {model_name}_{size} : {e}")
                all_results.append({
                    "run": f"{model_name}_{size}",
                    "error": str(e),
                })

    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    print_comparison_table([r for r in all_results if "MRR" in r])

    total = time.time() - pipeline_start
    print(f"\n{'='*55}")
    print(f"TERMINÉ en {total/60:.1f} min")
    print(f"Résultats → {results_path}")


if __name__ == "__main__":
    run()