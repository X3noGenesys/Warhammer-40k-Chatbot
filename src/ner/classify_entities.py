import json
import re
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from config import DATABRICKS_HOST, DATABRICKS_TOKEN


INPUT_PATH       = "data/title_entities.jsonl"
OUTPUT_PATH      = "data/title_entities.jsonl"
TEXT_INPUT_PATH  = "data/cleaned_data.jsonl"

BATCH_SIZE  = 30
MAX_RETRIES = 3
TIMEOUT     = 300
MAX_WORKERS = 6

WORKER_MODELS = [
    {"model_id": "databricks-meta-llama-3-3-70b-instruct",  "time_out_end": None},
    {"model_id": "databricks-qwen3-next-80b-a3b-instruct",  "time_out_end": None},
    {"model_id": "databricks-llama-4-maverick",             "time_out_end": None},
    {"model_id": "databricks-gpt-oss-120b",                 "time_out_end": None},
]

_model_lock = threading.Lock()

ENTITY_TYPES = [
    "CHARACTER",
    "FACTION",
    "LOCATION",
    "ARTIFACT",
    "EVENT",
    "ORGANIZATION",
    "CONCEPT",
]

TEXT_SNIPPET_LENGTH = 1000


SYSTEM_PROMPT = """\
Tu es un expert de l'univers Warhammer 40,000.
On te donne une liste d'entites W40k avec, pour chacune, un court extrait
du document wiki qui lui est consacre.
Pour chacune, indique le type parmi :

CHARACTER    : individus nommes (Horus, Guilliman, Inquisiteur Eisenhorn...)
FACTION      : races, armees, alliances (Orks, Tyranides, Empire T'au...)
LOCATION     : lieux, planetes, systemes (Terra, Cadia, Oeil de la Terreur...)
ARTIFACT     : objets, armes, reliques nommes (Trone d'Or, Epee des Heros...)
EVENT        : evenements historiques (Heresie d'Horus, Chute de Cadia...)
ORGANIZATION : chapitres, legions, ordres, institutions (Ultramarines, Adeptus Mechanicus...)
CONCEPT      : termes abstraits du lore (Warp, psyker, Ames des Anciens...)

Si l'entite n'appartient clairement a aucun type, utilise UNKNOWN.

Reponds UNIQUEMENT avec un JSON dans des balises <json_answer> :
<json_answer>
[
  {"name": "Horus", "type": "CHARACTER"},
  {"name": "Terra", "type": "LOCATION"}
]
</json_answer>\
"""

USER_TEMPLATE = """\
Classifie ces entites Warhammer 40,000 en te basant sur leur nom ET l'extrait de texte :

{names_list}\
"""

RETRY_TEMPLATE = (
    "Ta reponse ne contient que {found}/{total} entites. "
    "Classifie TOUTES les entites de la liste dans <json_answer></json_answer>."
)


def build_client() -> OpenAI:
    token    = DATABRICKS_TOKEN
    host     = DATABRICKS_HOST
    base_url = f"https://{host}/serving-endpoints"
    print(f"[CLIENT] Connecte a : {base_url}")
    return OpenAI(api_key=token, base_url=base_url)


def _release_expired():
    now = time.time()
    for m in WORKER_MODELS:
        if m["time_out_end"] and m["time_out_end"] < now:
            m["time_out_end"] = None


def get_model(worker_index: int) -> dict:
    model = WORKER_MODELS[worker_index % len(WORKER_MODELS)]
    while True:
        with _model_lock:
            _release_expired()
            if not model["time_out_end"]:
                return model
            wait = max(1, int(model["time_out_end"] - time.time()))
        time.sleep(wait)


def mark_timeout(model: dict):
    with _model_lock:
        model["time_out_end"] = time.time() + TIMEOUT


def get_response_text(response) -> str:
    content = response.choices[0].message.content
    if content is None:           return ""
    if isinstance(content, str):  return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if isinstance(b, dict) and "text" in b)
    return str(content)


def parse_classifications(
    raw: str,
    expected_names: list[str],
) -> dict[str, str]:
    match = re.search(
        r"<json_answer>\s*(.*?)\s*</json_answer>",
        raw, re.DOTALL | re.IGNORECASE
    )
    if not match:
        match2 = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match2:
            return {}
        candidate = match2.group(0)
    else:
        candidate = match.group(1).strip()

    try:
        data = json.loads(candidate)
    except Exception:
        return {}

    if not isinstance(data, list):
        return {}

    result = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        typ  = item.get("type", "UNKNOWN").strip().upper()
        if name and typ in ENTITY_TYPES + ["UNKNOWN"]:
            result[name.lower()] = typ

    expected_lower = {n.lower() for n in expected_names}
    missing = expected_lower - set(result.keys())
    if missing:
        sample = sorted(missing)[:5]
        print(f"    [WARN] {len(missing)} entite(s) absente(s) de la reponse : {sample}")

    return result

def classify_batch(
    client: OpenAI,
    name_snippet_pairs: list[tuple[str, str]],
    worker_index: int,
    batch_label: str,
) -> dict[str, str]:
    items = []
    for name, snippet in name_snippet_pairs:
        if snippet:
            items.append(f'- "{name}"\n  Extrait : {snippet}')
        else:
            items.append(f'- "{name}"')
    names_list = "\n\n".join(items)

    names_only = [name for name, _ in name_snippet_pairs]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(names_list=names_list)},
    ]

    raw = ""
    for attempt in range(1, MAX_RETRIES + 1):
        model = get_model(worker_index)
        print(f"  [{batch_label}] Tentative {attempt} — {model['model_id']}")
        try:
            response = client.chat.completions.create(
                model=model["model_id"],
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            raw    = get_response_text(response)
            result = parse_classifications(raw, names_only)

            if len(result) >= len(names_only) * 0.8:
                print(f"  [{batch_label}] OK — {len(result)}/{len(names_only)} classes")
                return result
            else:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": RETRY_TEMPLATE.format(
                        found=len(result),
                        total=len(names_only),
                    ),
                })

        except Exception as e:
            err = str(e)
            if "REQUEST_LIMIT_EXCEEDED" in err:
                mark_timeout(model)
            else:
                print(f"  [{batch_label}] Erreur : {err[:100]}")
                break

    return parse_classifications(raw, names_only)


def load_jsonl(path: str) -> list[dict]:
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


def run():
    from collections import Counter

    print("=" * 55)
    print("CLASSIFY ENTITIES — Classification des UNKNOWN")
    print("=" * 55)

    client   = build_client()
    entities = load_jsonl(INPUT_PATH)

    print(f"\n[INIT] Chargement des extraits de texte depuis {TEXT_INPUT_PATH}...")
    text_snippets: dict[str, str] = {}
    if Path(TEXT_INPUT_PATH).exists():
        with open(TEXT_INPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc     = json.loads(line)
                url     = doc.get("url", "")
                text    = doc.get("text", "")
                if url and text:
                    snippet     = text[:TEXT_SNIPPET_LENGTH].strip()
                    last_period = snippet.rfind(". ")
                    if last_period > TEXT_SNIPPET_LENGTH // 2:
                        snippet = snippet[:last_period + 1]
                    text_snippets[url] = snippet
        print(f"  {len(text_snippets):,} extraits charges.")
    else:
        print(f"  [WARN] {TEXT_INPUT_PATH} absent — classification sans texte.")

    unknowns = [e for e in entities if e["type"] == "UNKNOWN"]
    knowns   = [e for e in entities if e["type"] != "UNKNOWN"]

    print(f"\n[INIT] Entites totales   : {len(entities):,}")
    print(f"[INIT] Deja classifiees  : {len(knowns):,}")
    print(f"[INIT] UNKNOWN a classer : {len(unknowns):,}")

    if not unknowns:
        print("[INIT] Rien a faire.")
        return

    batches = [
        unknowns[i:i + BATCH_SIZE]
        for i in range(0, len(unknowns), BATCH_SIZE)
    ]
    print(f"[INIT] {len(batches)} batches de {BATCH_SIZE} entites\n")

    entity_index = {e["text"].lower(): e for e in unknowns}
    classified   = 0
    t0           = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                classify_batch,
                client,
                [(e["text"], text_snippets.get(e.get("source_url", ""), ""))
                 for e in batch],
                i % MAX_WORKERS,
                f"batch {i+1}/{len(batches)}",
            ): i
            for i, batch in enumerate(batches)
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                for name_lower, typ in result.items():
                    if name_lower in entity_index and typ != "UNKNOWN":
                        entity_index[name_lower]["type"] = typ
                        classified += 1
            except Exception as e:
                print(f"  [ERREUR] Batch : {e}")

            elapsed = time.time() - t0
            eta     = (elapsed / done) * (len(batches) - done) / 60
            print(f"[PROGRESS] {done}/{len(batches)} batches "
                  f"| {classified} entites classifiees "
                  f"| ETA ~{eta:.1f} min")

    all_entities = knowns + list(entity_index.values())

    type_counts   = Counter(e["type"] for e in all_entities)
    still_unknown = type_counts.get("UNKNOWN", 0)

    print(f"\n[STATS]")
    print(f"  Classes avec succes  : {classified:,}")
    print(f"  Encore UNKNOWN       : {still_unknown:,}")
    print(f"\n  Par type :")
    for t, n in type_counts.most_common():
        bar = "█" * (n * 25 // max(type_counts.values()))
        print(f"    {t:<15} {n:>5}  {bar}")

    save_jsonl(all_entities, OUTPUT_PATH)
    print(f"\n[SAVE] {OUTPUT_PATH} ({len(all_entities):,} entites)")
    print(f"Duree totale : {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    run()
