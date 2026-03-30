import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

MODELS_DIR  = Path("models")
DATA_DIR    = Path("data/kge")
REPORTS_DIR = Path("reports")

DATASET_SIZES = ["20k", "50k", "full"]
MODEL_NAMES   = ["transe", "rotate"]

NN_EXAMPLES = 5
NN_K        = 5

ENTITY_TYPE_COLORS = {
    "character":    "#E24B4A",
    "faction":      "#378ADD",
    "location":     "#1D9E75", 
    "organization": "#EF9F27",
    "event":        "#7F77DD",
    "artifact":     "#D85A30", 
    "concept":      "#888780",
}

MODEL_CONFIGS_LABELS = {"transe": "TransE", "rotate": "RotatE"}


def to_real_embeddings(emb: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(emb):
        return np.concatenate([emb.real, emb.imag], axis=1).astype(np.float32)
    return emb.astype(np.float32)


def load_model_data(model_name: str, size: str) -> dict | None:
    run_dir = MODELS_DIR / f"{model_name}_{size}"

    if not run_dir.exists():
        print(f"  [WARN] {run_dir} introuvable, ignoré.")
        return None

    try:
        entity_emb   = np.load(str(run_dir / "embeddings_entity.npy"))
        relation_emb = np.load(str(run_dir / "embeddings_relation.npy"))

        entity_labels = json.loads((run_dir / "entity_labels.json").read_text())
        rel_labels    = json.loads((run_dir / "relation_labels.json").read_text())

        results_path = MODELS_DIR / "results.json"
        metrics = {}
        if results_path.exists():
            all_results = json.loads(results_path.read_text())
            run_key = f"{model_name}_{size}"
            for r in all_results:
                if r.get("run") == run_key and "MRR" in r:
                    metrics = {
                        "MRR":     r["MRR"],
                        "Hits@1":  r["Hits@1"],
                        "Hits@3":  r["Hits@3"],
                        "Hits@10": r["Hits@10"],
                    }
                    break

        return {
            "entity_emb":    entity_emb,
            "relation_emb":  relation_emb,
            "entity_labels": entity_labels,
            "rel_labels":    rel_labels,
            "metrics":       metrics,
            "run_dir":       run_dir,
        }

    except Exception as e:
        print(f"  [ERREUR] Chargement {model_name}_{size} : {e}")
        return None

def plot_metrics(all_data: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Knowledge Graph Embeddings — Métriques", fontsize=14, fontweight="bold")

    sizes_label = DATASET_SIZES
    colors      = {"transe": "#378ADD", "rotate": "#E24B4A"}
    markers     = {"transe": "o",       "rotate": "s"}

    ax = axes[0]
    ax.set_title("MRR par taille de dataset")
    ax.set_xlabel("Taille du dataset")
    ax.set_ylabel("MRR (Mean Reciprocal Rank)")

    x_pos = range(len(sizes_label))
    for model_name in MODEL_NAMES:
        mrr_vals = []
        for size in DATASET_SIZES:
            key  = f"{model_name}_{size}"
            data = all_data.get(key)
            mrr  = data["metrics"].get("MRR", 0.0) if data else 0.0
            mrr_vals.append(mrr)

        ax.plot(x_pos, mrr_vals,
                color=colors[model_name],
                marker=markers[model_name],
                linewidth=2, markersize=8,
                label=MODEL_CONFIGS_LABELS.get(model_name, model_name))
        for x, y in zip(x_pos, mrr_vals):
            if y > 0:
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(sizes_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax2  = axes[1]
    ax2.set_title("Hits@k — dataset complet")
    ax2.set_xlabel("Modèle")
    ax2.set_ylabel("Score")

    hit_metrics = ["Hits@1", "Hits@3", "Hits@10"]
    hit_colors  = ["#1D9E75", "#EF9F27", "#7F77DD"]
    x2_pos      = np.arange(len(MODEL_NAMES))
    width       = 0.25

    for i, (metric, color) in enumerate(zip(hit_metrics, hit_colors)):
        vals = []
        for model_name in MODEL_NAMES:
            key  = f"{model_name}_full"
            data = all_data.get(key)
            val  = data["metrics"].get(metric, 0.0) if data else 0.0
            vals.append(val)

        bars = ax2.bar(x2_pos + i * width, vals, width,
                       label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    model_labels = [MODEL_CONFIGS_LABELS.get(m, m) for m in MODEL_NAMES]
    ax2.set_xticks(x2_pos + width)
    ax2.set_xticklabels(model_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    out = REPORTS_DIR / "kge_metrics.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Métriques → {out}")


def get_entity_type(slug: str) -> str:
    parts = slug.split("_")
    if parts[0] in ENTITY_TYPE_COLORS:
        return parts[0]
    return "concept"


def plot_tsne(model_name: str, size: str, data: dict):
    entity_labels = data["entity_labels"]

    entity_emb = to_real_embeddings(data["entity_emb"])

    n = len(entity_emb)
    print(f"  t-SNE sur {n:,} entités (dim={entity_emb.shape[1]})...")

    perplexity = min(30, n // 4)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        learning_rate="auto",
        init="pca",
    )
    t0        = time.time()
    coords_2d = tsne.fit_transform(entity_emb)
    print(f"  t-SNE terminé en {time.time()-t0:.1f}s")

    colors = []
    for idx in range(n):
        slug  = entity_labels.get(str(idx), "")
        etype = get_entity_type(slug)
        colors.append(ENTITY_TYPE_COLORS.get(etype, "#888780"))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"t-SNE — {MODEL_CONFIGS_LABELS.get(model_name, model_name)} ({size})",
                 fontsize=13, fontweight="bold")

    ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
               c=colors, s=6, alpha=0.6, linewidths=0)

    notable = [
        "character_horus", "character_roboute_guilliman",
        "faction_space_marines", "faction_chaos",
        "location_terra", "event_heresie_d_horus",
        "organization_ultramarines", "organization_death_guard",
        "concept_warp", "concept_psyker",
    ]
    for slug, idx_str in entity_labels.items():
        if slug in notable:
            idx = int(idx_str)
            ax.annotate(
                slug.split("_", 1)[-1].replace("_", " ").title(),
                (coords_2d[idx, 0], coords_2d[idx, 1]),
                fontsize=7,
                xytext=(5, 5), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    legend_patches = [
        mpatches.Patch(color=color, label=etype.capitalize())
        for etype, color in ENTITY_TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, framealpha=0.8)
    ax.axis("off")

    out = REPORTS_DIR / f"tsne_{model_name}_{size}.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE sauvegardé → {out}")


def nearest_neighbors(model_name: str, size: str, data: dict) -> list[dict]:
    entity_labels = data["entity_labels"]

    entity_emb = to_real_embeddings(data["entity_emb"])

    slug_to_id = {v: int(k) for k, v in entity_labels.items()}

    seed_slugs = [
        "character_horus",
        "character_roboute_guilliman",
        "faction_space_marines",
        "faction_chaos",
        "location_terra",
        "organization_ultramarines",
        "concept_warp",
    ]

    norms   = np.linalg.norm(entity_emb, axis=1, keepdims=True)
    normed  = entity_emb / (norms + 1e-8)
    sim_mat = normed @ normed.T

    results = []
    for slug in seed_slugs:
        if slug not in slug_to_id:
            continue

        idx     = slug_to_id[slug]
        sims    = sim_mat[idx]

        top_ids = np.argsort(sims)[::-1][1:NN_K + 1]

        neighbors = []
        for nid in top_ids:
            neighbor_slug = entity_labels.get(str(nid), str(nid))
            neighbors.append({
                "slug":       neighbor_slug,
                "similarity": round(float(sims[nid]), 4),
            })

        results.append({
            "query":     slug,
            "neighbors": neighbors,
        })

    return results


def print_nearest_neighbors(nn_results: list[dict], model_name: str, size: str):
    print(f"\n  Plus proches voisins — {MODEL_CONFIGS_LABELS.get(model_name, model_name)} ({size}) :")
    for item in nn_results[:NN_EXAMPLES]:
        query = item["query"].split("_", 1)[-1].replace("_", " ").title()
        print(f"\n    [{query}]")
        for n in item["neighbors"]:
            name = n["slug"].split("_", 1)[-1].replace("_", " ").title()
            print(f"      {n['similarity']:.4f}  {name}")


def build_report(all_data: dict, all_nn: dict) -> dict:
    report = {
        "metrics_by_run":    {},
        "sensitivity_table": {},
        "nearest_neighbors": {},
    }

    for key, data in all_data.items():
        if data and data.get("metrics"):
            report["metrics_by_run"][key] = data["metrics"]

    for model_name in MODEL_NAMES:
        report["sensitivity_table"][model_name] = {}
        for size in DATASET_SIZES:
            key  = f"{model_name}_{size}"
            data = all_data.get(key)
            report["sensitivity_table"][model_name][size] = (
                data["metrics"] if data and data.get("metrics") else {}
            )

    report["nearest_neighbors"] = all_nn
    return report


def run():
    print("=" * 55)
    print("KGE EVAL — Évaluation et visualisations")
    print("=" * 55)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_data: dict = {}
    for model_name in MODEL_NAMES:
        for size in DATASET_SIZES:
            key           = f"{model_name}_{size}"
            all_data[key] = load_model_data(model_name, size)

    print(f"\n{'='*65}")
    print(f"{'Run':<22} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}")
    print("-" * 65)
    for model_name in MODEL_NAMES:
        for size in DATASET_SIZES:
            key  = f"{model_name}_{size}"
            data = all_data.get(key)
            if data and data.get("metrics"):
                m = data["metrics"]
                print(f"  {key:<20} {m.get('MRR',0):>8.4f} "
                      f"{m.get('Hits@1',0):>8.4f} "
                      f"{m.get('Hits@3',0):>8.4f} "
                      f"{m.get('Hits@10',0):>8.4f}")
            else:
                print(f"  {key:<20} {'N/A':>8}")

    print("\n[PLOT] Génération des graphiques de métriques...")
    plot_metrics(all_data)

    print("\n[TSNE] Génération des visualisations t-SNE (dataset full)...")
    for model_name in MODEL_NAMES:
        key  = f"{model_name}_full"
        data = all_data.get(key)
        if data:
            plot_tsne(model_name, "full", data)
        else:
            print(f"  [SKIP] {key} non disponible")

    print("\n[NN] Calcul des plus proches voisins...")
    all_nn = {}
    for model_name in MODEL_NAMES:
        key  = f"{model_name}_full"
        data = all_data.get(key)
        if data:
            nn_results = nearest_neighbors(model_name, "full", data)
            print_nearest_neighbors(nn_results, model_name, "full")
            all_nn[key] = nn_results

    report      = build_report(all_data, all_nn)
    report_path = REPORTS_DIR / "kge_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] Rapport → {report_path}")

    print(f"\n{'='*55}")
    print(f"TERMINÉ en {time.time()-t0:.1f}s")
    print(f"\nFichiers produits :")
    for f in sorted(REPORTS_DIR.glob("kge_*")):
        print(f"  {f}")


if __name__ == "__main__":
    run()