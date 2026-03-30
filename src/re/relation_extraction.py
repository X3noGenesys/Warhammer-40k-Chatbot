import json
import re
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATABRICKS_HOST, DATABRICKS_TOKEN
from openai import OpenAI


TEXT_INPUT_PATH   = "data/cleaned_data.jsonl"
NER_INPUT_PATH    = "data/merged_ner.jsonl"
OUTPUT_PATH       = "data/relations.jsonl"

CLEAN_OUTPUT       = False
NB_DOCS_TO_PROCESS = None

DELAY       = 2
TIMEOUT     = 300
MAX_RETRIES = 3
CHUNK_SIZE  = 6000
CHUNK_OVERLAP = 300

RELATION_TYPES = [
    "EST_MEMBRE_DE",
    "DIRIGE",
    "EST_ENNEMI_DE",
    "EST_ALLIE_DE",
    "A_PARTICIPE_A",
    "S_EST_DEROULE_A",
    "EST_SITUE_A",
    "MANIE",
    "A_CREE",
    "VENERE",
    "CONTROLE",
    "EST_SOUS_GROUPE_DE",
    "EST_ORIGINAIRE_DE",
    "A_VAINCU",
    "A_TRAHI",
]


WORKER_MODELS = [
    {"model_id": "databricks-gpt-oss-120b",                 "time_out_end": None},
    {"model_id": "databricks-llama-4-maverick",             "time_out_end": None},
    {"model_id": "databricks-meta-llama-3-3-70b-instruct",  "time_out_end": None},
    {"model_id": "databricks-qwen3-next-80b-a3b-instruct",  "time_out_end": None},
]

MAX_WORKERS = len(WORKER_MODELS)
_model_lock = threading.Lock()
_write_lock = threading.Lock()


SYSTEM_PROMPT = """\
Tu es un système d'extraction de relations spécialisé dans l'univers Warhammer 40,000.

## Ta tâche
À partir du texte fourni et de la liste d'entités connues, extraire toutes les relations
explicites entre ces entités.

## Types de relations autorisés
- EST_MEMBRE_DE      : une entité appartient à une organisation ou faction
- DIRIGE             : un personnage commande ou dirige une entité
- EST_ENNEMI_DE      : antagonisme, guerre, conflit entre deux entités
- EST_ALLIE_DE       : alliance, coopération, loyauté entre deux entités
- A_PARTICIPE_A      : participation à un événement (bataille, croisade…)
- S_EST_DEROULE_A    : un événement a eu lieu dans un lieu précis
- EST_SITUE_A        : une entité (organisation, lieu) est localisée dans un lieu
- MANIE              : un personnage utilise ou porte un artefact nommé
- A_CREE             : création d'un artefact ou d'une entité
- VENERE             : culte ou vénération envers une entité ou un concept
- CONTROLE           : contrôle territorial ou politique d'un lieu
- EST_SOUS_GROUPE_DE : sous-organisation, chapitre dérivé d'une entité plus grande
- EST_ORIGINAIRE_DE  : origine géographique ou factionnelle d'un personnage
- A_VAINCU           : victoire militaire ou personnelle sur une entité
- A_TRAHI            : acte de trahison envers une entité

## Règles
- N'extraire que les relations clairement exprimées dans le texte.
- Les champs "source" et "target" doivent contenir le nom exact de l'entité tel qu'il apparaît.
- Ne pas inventer de relations absentes du texte.
- Si aucune relation n'est trouvée, retourner un tableau vide.

## Format de sortie
Raisonne librement, puis encapsule ta réponse finale dans des balises <json_answer> :

<json_answer>
[
  {"source": "Horus", "relation": "EST_MEMBRE_DE", "target": "Luna Wolves"},
  {"source": "Horus", "relation": "A_TRAHI", "target": "Empereur de l'Humanité"},
  {"source": "Bataille de Terra", "relation": "S_EST_DEROULE_A", "target": "Terra"}
]
</json_answer>

Le contenu entre les balises <json_answer> doit être un tableau JSON valide et rien d'autre.\
"""

USER_PROMPT_TEMPLATE = """\
Extrait toutes les relations Warhammer 40,000 présentes dans le texte suivant.

Entités connues présentes dans ce texte (utilise ces noms exacts) :
{entities_list}

<text>
{text}
</text>\
"""

def build_client() -> OpenAI:
    token    = DATABRICKS_TOKEN
    host     = DATABRICKS_HOST

    base_url = f"https://{host}/serving-endpoints"
    print(f"[CLIENT] Connecté à : {base_url}")
    return OpenAI(api_key=token, base_url=base_url)


def _release_expired_timeouts():
    now = time.time()
    for m in WORKER_MODELS:
        if m["time_out_end"] and m["time_out_end"] < now:
            print(f"  [MODEL] Timeout expiré pour '{m['model_id']}'.")
            m["time_out_end"] = None


def get_assigned_model(worker_index: int) -> dict:
    model = WORKER_MODELS[worker_index]
    while True:
        with _model_lock:
            _release_expired_timeouts()
            if not model["time_out_end"]:
                return model
            wait = max(1, int(model["time_out_end"] - time.time()))
        print(f"  [MODEL] Worker {worker_index} en timeout, attente {wait}s...")
        time.sleep(wait)


def get_any_available_model(exclude_index: int) -> dict | None:
    with _model_lock:
        _release_expired_timeouts()
        available = [
            m for i, m in enumerate(WORKER_MODELS)
            if i != exclude_index and not m["time_out_end"]
        ]
        return available[0] if available else None


def mark_model_timeout(model: dict):
    with _model_lock:
        model["time_out_end"] = time.time() + TIMEOUT
    print(f"  [MODEL] '{model['model_id']}' mis en quarantaine {TIMEOUT}s.")


def get_response_text(response) -> str:
    content = response.choices[0].message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def extract_json_from_response(raw: str) -> list[dict]:
    tag_match = re.search(
        r"<json_answer>\s*(.*?)\s*</json_answer>",
        raw, flags=re.DOTALL | re.IGNORECASE,
    )
    if tag_match:
        return _parse_json_array(tag_match.group(1).strip(), source="<json_answer>")
    md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if md_match:
        return _parse_json_array(md_match.group(1).strip(), source="```json```")
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return _parse_json_array(raw[start:end + 1], source="bracket extraction")
    raise ValueError("No JSON array found in LLM response.")


def _parse_json_array(candidate: str, source: str) -> list[dict]:
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"[{source}] Invalid JSON: {e} | Content: {candidate[:200]}")
    if not isinstance(data, list):
        raise ValueError(f"[{source}] Expected array, got {type(data).__name__}")
    return data


def validate_relations(relations: list[dict]) -> list[dict]:
    valid = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        source   = rel.get("source", "").strip()
        relation = rel.get("relation", "").strip().upper()
        target   = rel.get("target", "").strip()
        if not source or not target or not relation:
            continue
        if relation not in RELATION_TYPES:
            continue
        valid.append({
            "source":   source,
            "relation": relation,
            "target":   target,
        })
    return valid


def deduplicate_relations(relations: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for rel in relations:
        key = (rel["source"].lower(), rel["relation"], rel["target"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            boundary = text.rfind(". ", max(start, end - 400), end)
            if boundary != -1:
                end = boundary + 1
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def extract_relations_from_chunk(
    client: OpenAI,
    text: str,
    entities_list: str,
    worker_index: int,
    chunk_label: str,
) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
            text=text,
            entities_list=entities_list,
        )},
    ]

    raw = ""
    attempts_done = 0

    while attempts_done < MAX_RETRIES:
        model = get_assigned_model(worker_index)
        attempts_done += 1
        print(f"    [{chunk_label}] Tentative {attempts_done}/{MAX_RETRIES} — {model['model_id']}")

        try:
            response = client.chat.completions.create(
                model=model["model_id"],
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )
            raw = get_response_text(response)
            relations = extract_json_from_response(raw)
            validated = validate_relations(relations)
            print(f"    [{chunk_label}] OK — {len(validated)} relations.")
            return validated

        except ValueError as parse_error:
            print(f"    [{chunk_label}] Parsing échoué (tentative {attempts_done}) : {parse_error}")
            if attempts_done < MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Ta réponse n'a pas pu être analysée. Erreur : {parse_error}\n\n"
                        "Réessaie. Ta réponse DOIT se terminer par :\n"
                        "<json_answer>\n[ ... ]\n</json_answer>"
                    ),
                })
                time.sleep(1)

        except Exception as api_error:
            error_str = str(api_error)
            print(f"    [{chunk_label}] Erreur API : {error_str}")
            if "REQUEST_LIMIT_EXCEEDED" in error_str:
                mark_model_timeout(model)
                fallback = get_any_available_model(exclude_index=worker_index)
                if fallback:
                    print(f"    [{chunk_label}] Bascule sur fallback : {fallback['model_id']}")
                    try:
                        response = client.chat.completions.create(
                            model=fallback["model_id"],
                            messages=messages,
                            temperature=0.0,
                            max_tokens=2048,
                        )
                        raw = get_response_text(response)
                        relations = extract_json_from_response(raw)
                        validated = validate_relations(relations)
                        print(f"    [{chunk_label}] Fallback OK — {len(validated)} relations.")
                        return validated
                    except Exception as fallback_error:
                        print(f"    [{chunk_label}] Fallback échoué : {fallback_error}")
                        attempts_done -= 1
                else:
                    attempts_done -= 1
                    time.sleep(5)
            else:
                break

    print(f"    [{chunk_label}] Abandon après {MAX_RETRIES} tentatives.")
    return []


def process_document(
    client: OpenAI,
    text: str,
    entities: list[dict],
    doc_index: int,
    total_docs: int,
    title: str,
) -> list[dict]:
    chunks = chunk_text(text)

    entities_list = "\n".join(
        f"- {e['text']} ({e['type']})"
        for e in entities[:60]
    ) or "(aucune entité connue)"

    print(f"\n[DOC {doc_index}/{total_docs}] '{title}'")
    print(f"  {len(text):,} chars → {len(chunks)} chunks | {len(entities)} entités connues")

    t0 = time.time()
    all_relations: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                extract_relations_from_chunk,
                client,
                chunk,
                entities_list,
                i % MAX_WORKERS,
                f"chunk {i+1}/{len(chunks)} → {WORKER_MODELS[i % MAX_WORKERS]['model_id'].split('databricks-')[1]}",
            ): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            try:
                all_relations.extend(future.result())
            except Exception as e:
                print(f"    [chunk {futures[future]+1}] Exception : {e}")

    deduped = deduplicate_relations(all_relations)
    elapsed = time.time() - t0
    print(f"  Terminé en {elapsed:.0f}s — "
          f"{len(all_relations)} relations brutes → {len(deduped)} après déduplication.")
    return deduped


def load_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_ner_index(ner_path: str) -> dict[str, list[dict]]:
    index = {}
    for doc in load_jsonl(ner_path):
        index[doc.get("url", "")] = doc.get("entities", [])
    return index


def get_already_processed_urls(output_path: str) -> set[str]:
    if not Path(output_path).exists():
        return set()
    return {doc.get("url", "") for doc in load_jsonl(output_path)}


def print_stats(output_path: str):
    from collections import Counter
    rel_counter: Counter = Counter()
    total_docs = total_rels = 0
    for doc in load_jsonl(output_path):
        total_docs += 1
        for rel in doc.get("relations", []):
            rel_counter[rel["relation"]] += 1
            total_rels += 1
    print(f"\n{'='*55}")
    print(f"STATISTIQUES RE")
    print(f"{'='*55}")
    print(f"Documents : {total_docs}  |  Relations : {total_rels}")
    print(f"\nPar type :")
    for t, n in rel_counter.most_common():
        print(f"  {t:<25} {n:>5}")


def run():
    print("=" * 55)
    print("RE PIPELINE — Warhammer 40k")
    print(f"CHUNK_SIZE={CHUNK_SIZE} | MAX_WORKERS={MAX_WORKERS}")
    print("=" * 55)

    client    = build_client()
    ner_index = build_ner_index(NER_INPUT_PATH)
    print(f"[INIT] Index NER chargé : {len(ner_index)} documents.")

    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUT and output_file.exists():
        output_file.unlink()

    all_docs = load_jsonl(TEXT_INPUT_PATH)
    if NB_DOCS_TO_PROCESS is not None:
        all_docs = all_docs[:NB_DOCS_TO_PROCESS]

    already_done = get_already_processed_urls(OUTPUT_PATH)
    to_process   = [d for d in all_docs if d.get("url", "") not in already_done]
    print(f"[INIT] {len(to_process)} documents à traiter (sur {len(all_docs)} total).")

    if not to_process:
        print("[INIT] Rien à faire.")
        print_stats(OUTPUT_PATH)
        return

    total_relations = 0
    pipeline_start  = time.time()

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, doc in enumerate(to_process, start=1):
            url   = doc.get("url", "")
            title = doc.get("title", "(sans titre)")
            text  = doc.get("text", "")

            try:
                entities  = ner_index.get(url, [])
                relations = process_document(client, text, entities, i, len(to_process), title)

                result = {
                    "url":       url,
                    "title":     title,
                    "relations": relations,
                    "stats":     {"relations_found": len(relations)},
                }
                with _write_lock:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()

                total_relations += len(relations)

                elapsed  = time.time() - pipeline_start
                eta_min  = (elapsed / i) * (len(to_process) - i) / 60
                print(f"  [PROGRESSION] {i}/{len(to_process)} — ETA ~{eta_min:.0f} min")

            except Exception as e:
                print(f"\n[ERREUR] Document {i} ignoré : {e}")

            if i < len(to_process):
                time.sleep(DELAY)

    elapsed_total = time.time() - pipeline_start
    print(f"\n{'='*55}")
    print(f"TERMINÉ en {elapsed_total/60:.1f} min")
    print(f"Relations extraites : {total_relations}")
    print()
    print_stats(OUTPUT_PATH)

if __name__ == "__main__":
    run()