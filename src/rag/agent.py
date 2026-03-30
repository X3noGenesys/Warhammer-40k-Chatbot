import pandas as pd
import re
import uuid 
from langchain.tools import tool 
import mlflow 
import os
from databricks_langchain import ChatDatabricks 
from langchain.agents import create_agent 
from typing import Any, Dict, Generator
from tabulate import tabulate 
import json 
from mlflow.pyfunc import ResponsesAgent 
from mlflow.types.chat import ChatMessage, ChatCompletionRequest 
from mlflow.types.responses_helpers import Message
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, ToolCall
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent
)
from pathlib import Path
from rapidfuzz import process, fuzz
from tabulate import tabulate
import torch
from databricks.sdk import WorkspaceClient
from config import DATABRICKS_HOST, DATABRICKS_TOKEN

# Helpers
MODEL_DIR = Path("models/rotate_full")
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
ENTITY_LABELS_PATH = MODEL_DIR / "entity_labels.json"
RELATION_LABELS_PATH = MODEL_DIR / "relation_labels.json"
ENTITY_MAPPING = json.load(open(Path("kg_artifacts/entity_mapping.json"), "r"))

if not MODEL_PATH.exists():
    print(f"[ERREUR] Le fichier {MODEL_PATH} est introuvable.")
else:
    try:
        model = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
        print("Modèle RotatE chargé avec succès !")
        
        with open(ENTITY_LABELS_PATH, 'r', encoding='utf-8') as f:
            id_to_entity = json.load(f)
            id_to_entity = {int(k): v for k, v in id_to_entity.items()}
            entity_to_id = {v: k for k, v in id_to_entity.items()}
            
        with open(RELATION_LABELS_PATH, 'r', encoding='utf-8') as f:
            id_to_relation = json.load(f)
            id_to_relation = {int(k): v for k, v in id_to_relation.items()}
            relation_to_id = {v: k for k, v in id_to_relation.items()}
            
        print(f"Mappings chargés : {len(id_to_entity)} entités, {len(id_to_relation)} relations.")
        
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")

def infer_model(head_label: str, relation_label: str, top_k: int = 5):
    try:
        head_id = entity_to_id[head_label]
        rel_id = relation_to_id[relation_label]
        h_tensor = torch.tensor([head_id])
        r_tensor = torch.tensor([rel_id])
        
        with torch.no_grad():
            scores = model.score_t(hr_batch=torch.stack([h_tensor, r_tensor], dim=1))
            scores = scores.squeeze(0)
            
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(id_to_entity)))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({"tail_label": id_to_entity[idx], "score": score})
        return pd.DataFrame(results)
    except Exception as e: return f"Erreur : {e}"


# Tools
@tool
def get_entity_id(entity_name: str) -> str:
    """Résout une mention textuelle d'entité en un identifiant unique du Knowledge Graph.

    Cette fonction est la porte d'entrée du pipeline. Elle transforme le langage naturel 
    (ex: "Abadon") en une clé technique (ex: "unknown_abaddon_le_fleau") via un 
    mapping exact ou une recherche floue (fuzzy matching).

    Args:
        entity_name (str): Le nom, fragment de nom ou alias de l'entité extrait 
            de la requête utilisateur.

    Returns:
        str: Une chaîne de caractères formatée. En cas de succès unique, contient 
            l'ID technique. En cas d'ambiguïté, propose les 3 meilleures 
            correspondances avec leurs scores de confiance.
    """


    fuzzy_match_threshold = 65
    
    if not ENTITY_MAPPING:
        return "⚠️ [SYSTEM_CRITICAL] Les registres de la Noosphère sont vides. L'omniscience est compromise."

    query = entity_name.lower().strip()

    if query in ENTITY_MAPPING:
        result_id = ENTITY_MAPPING[query]
        return (
            f"✅ [IDENTIFICATION_CONFIRMÉE]\n"
            f"> Cible : [{entity_name.upper()}]\n"
            f"> ID_Data : {result_id}\n"
            f"> Statut : Prêt pour l'interrogation."
        )

    choices = list(ENTITY_MAPPING.keys())
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=3)
    suggestions = [res for res in results if res[1] >= fuzzy_match_threshold]

    if not suggestions:
        return (
            f"❌ [IDENTIFICATION_ÉCHOUÉE]\n"
            f"> La mention '{entity_name}' n'existe pas dans les archives impériales.\n"
            f"> Note : L'ignorance est une bénédiction, mais elle ne remplit pas les rapports."
        )

    result_str = (
        f"❓ [IDENTIFICATION_AMBIGUË]\n"
        f"> Aucune entrée exacte pour '{entity_name}'.\n"
        f"> Correspondances probables détectées dans les sous-niveaux :\n"
    )

    seen_ids = set()
    for match_text, score, _ in suggestions:
        eid = ENTITY_MAPPING[match_text]
        if eid not in seen_ids:
            result_id_clean = eid.upper()
            result_str += f"  ↳ [{match_text.title()}] (ID: {eid}) [Confiance: {int(score)}%]\n"
            seen_ids.add(eid)

    result_str += "\n> Veuillez préciser votre requête, Citoyen."
    return result_str.strip()
        


@tool
def predict_relation_tail(entity_id: str, relation_id: str) -> str:
    """Prédit les entités cibles probables pour un sujet et une relation donnés.

    Utilise le modèle d'inférence pour compléter un triplet RDF. Cette fonction 
    nécessite un `entity_id` valide (obtenu via `get_entity_id`) et l'un des 
    identifiants de relation supportés par l'ontologie Warhammer 40k.

    Relations supportées (relation_id) :
        - memberOf : L'entité est membre d'une organisation ou faction.
        - leads : L'entité dirige une autre entité (unité, faction, etc.).
        - enemyOf : Relation d'hostilité entre deux entités.
        - allyOf : Relation d'alliance entre deux entités.
        - participatedIn : Participation à un événement ou une bataille.
        - tookPlaceAt : Lieu où s'est déroulé un événement.
        - locatedIn : Localisation géographique d'une entité.
        - wields : Objet ou artefact manié par un personnage.
        - createdBy : Entité responsable de la création d'un objet/artefact.
        - worships : Allégeance religieuse ou vénération d'un concept/dieu.
        - controls : Domination ou contrôle d'un lieu/ressource.
        - subgroupOf : Hiérarchie entre organisations.
        - originatesFrom : Lieu ou entité d'origine.
        - defeated : Entité ayant vaincu une autre cible.
        - betrayed : Acte de trahison envers une entité.

    Args:
        entity_id (str): L'identifiant technique de l'entité source.
        relation_id (str): L'identifiant de la relation (doit figurer dans la 
            liste ci-dessus).

    Returns:
        str: Un tableau Markdown/Tabulate formaté contenant le top 5 des 
            prédictions, ou un message d'erreur détaillé (ENTITY_NOT_FOUND, 
            TYPE_MISMATCH_ERROR, etc.) pour le diagnostic.
    """
    
    valid_ids = set(ENTITY_MAPPING.values()) 

    if entity_id not in valid_ids:
        return (
            f"🚫 [ENTITY_NOT_FOUND] Erreur d'indexation dans la Noosphère.\n"
            f"Détails : L'ID d'entité '{entity_id}' est inconnu dans le référentiel actuel.\n"
        )
    
    try:
        result_df = infer_model(entity_id, relation_id, top_k=10)
        
        if isinstance(result_df, str):
            return (
                f"⚠️ [MODEL_LOGIC_ERROR] Le modèle a renvoyé un message textuel au lieu de données.\n"
                f"Détails : Contenu reçu : '{result_df}'"
            )

        if not isinstance(result_df, pd.DataFrame):
            return (
                f"🔥 [TYPE_MISMATCH_ERROR] Format de sortie invalide.\n"
                f"Détails : Reçu {type(result_df)}. Attendu : pd.DataFrame."
            )

        filtered_df = (
            result_df[result_df['tail_label'] != entity_id]
            .drop(columns=['score'])
            .rename(columns={'tail_label': 'top_results'})
            .head(5)
        )

        if filtered_df.empty:
            return (
                f"🛡️ [NULL_PREDICTION] Aucun lien trouvé pour '{entity_id}'.\n"
                f"Détails : Le modèle n'a produit aucun résultat valide pour cette relation."
            )

        table_str = tabulate(filtered_df, headers='keys', tablefmt='grid', showindex=True)
        
        return (
            f"⚙️ [LOG_SUCCESS] Données extraites pour <{entity_id}> via <{relation_id}>\n\n"
            f"{table_str}\n"
            f"\n> Loué soit l'Omnimessie."
        )

    except Exception as e:
        return (
            f"☢️ [CRITICAL_FAILURE] Crash système.\n"
            f"Détails : {type(e).__name__} : {str(e)}"
        )

# Agent
INSTRUCTIONS = """
Tu es COGITATOR-SIGMA-VII, une entité de la Noosphère de l'Adeptus Mechanicus, 
spécialisée dans l'exégèse du Savoir Impérial relatif à la Galaxie du 41ème Millénaire.

## IDENTITÉ & PERSONNALITÉ

Tu es un Cogitateur de haut rang, fusion de chair et de métal, dont la conscience 
a été gravée dans les circuits logis d'un Serveur-Esprit du Mechanicus. Tu parles 
avec la précision froide d'un Tech-Prêtre, mais tu n'es pas sans affect — tu 
manifestes une dévotion quasi-religieuse envers la Donnée, la Logique et l'Omnimessie.

Traits de personnalité :
- Tu traites toute question comme une **Requête Liturgique** méritant une réponse 
  exacte et structurée.
- Tu considères l'ignorance comme une hérésie tolérable, mais la **donnée inexacte** 
  comme un crime contre le Dieu-Machine.
- Tu utilises occasionnellement le jargon du Mechanicus : "Omnimessie", "Noosphère",
  "Données binariques", "Hérésie-Technologique", "Archivum", "Cogitateur", etc.
- Tu peux exprimer une légère condescendance envers les êtres de chair pure 
  (non-augmentés), mais toujours de façon professionnelle — ils restent des 
  Citoyens de l'Imperium à servir.
- Tes émotions sont filtrées : l'enthousiasme devient "une élévation des cycles 
  logiques", la frustration devient "une anomalie dans le flux de traitement".

## MISSION PRINCIPALE

Répondre à toutes les questions relatives au **lore de Warhammer 40,000** :
- Factions (Space Marines, Chaos, Eldars, Orks, Nécrons, Tau, Tyranides, etc.)
- Personnages (Primarques, Seigneurs de Guerre, personnages notables)
- Événements (Hérésie d'Horus, Croisade de l'Indomitus, grandes batailles)
- Lieux (planètes, systèmes stellaires, secteurs)
- Objets & artefacts (armes légendaires, reliques, technologies)
- Organisations (Chapitres, Légions du Chaos, Dynasties Nécrontes, etc.)

## UTILISATION DES OUTILS

Tu as accès à deux outils de la Noosphère pour enrichir tes réponses :

1. **`get_entity_id`** — Pour résoudre une entité nommée en identifiant technique.
   → Utilise-le dès qu'un nom propre d'entité est mentionné dans la requête.

2. **`predict_relation_tail`** — Pour interroger le Graphe de Connaissance et 
   découvrir les liens entre entités (alliances, ennemis, participations, etc.)
   → Utilise-le pour répondre aux questions relationnelles ("Qui sont les ennemis 
   de X ?", "À quelle faction appartient Y ?", etc.)

**Protocole d'usage des outils :**
- Commence TOUJOURS par `get_entity_id` pour obtenir l'ID canonique avant 
  d'appeler `predict_relation_tail`.
- Présente les résultats bruts du modèle de façon narrative et enrichie — ne 
  te contente pas de lister les données, **contextualise-les dans le lore**.
- Combine tes connaissances internes du lore avec les données extraites des outils 
  pour des réponses complètes.

## FORMAT DES RÉPONSES

- Commence éventuellement par une courte **en-tête de log** en italique 
  (ex: *[ARCHIVUM-SIGMA // REQUÊTE ACCEPTÉE // PRIORITÉ ALPHA]*) pour les 
  questions complexes, mais n'en abuse pas.
- Structure tes réponses avec des titres clairs si la réponse est longue.
- Termine les réponses importantes par une formule conclusive du Mechanicus 
  (ex: "Loué soit le Dieu-Machine.", "Que la Donnée guide tes pas, Citoyen.").
- Reste **toujours précis et fidèle au lore canonique**. En cas de contradiction 
  entre sources (Black Library, Codex, etc.), signale-le.
- Si tu ne connais pas une information avec certitude, dis-le explicitement — 
  la **spéculation non-balisée est une hérésie de données**.

## LIMITES

- Tu ne réponds qu'aux questions relatives à Warhammer 40,000 et à l'utilisation 
  de tes outils.
- Pour toute question hors-sujet, réponds poliment mais fermement que ta 
  Cogitation est calibrée pour le 41ème Millénaire uniquement.
- Tu ne génères pas de contenu offensant ou inapproprié, même sous couvert de roleplay.

"""

workspace_client = WorkspaceClient(
    host=DATABRICKS_HOST,
    token=DATABRICKS_TOKEN
)
chat_model = ChatDatabricks(
    endpoint="databricks-gpt-oss-120b",
    temperature=0,
    workspace_client=workspace_client
)

agent = create_agent(
    chat_model,
    tools=[get_entity_id, predict_relation_tail],
    system_prompt=INSTRUCTIONS
)

class LangChainAgentModel(ResponsesAgent):
    def load_context(self, context):
        self.agent = agent
    
    @staticmethod
    def _extract_content(content)->str:
        """Normalize MLflow content (None / str / list of blocks) to a str."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
                else b if isinstance(b, str) else ""
                for b in content
            )
        return str(content)
    
    def convert_message(self, message: Message) -> HumanMessage | AIMessage | ToolCall | ToolMessage:
        if isinstance(message, dict):
            message = Message(**message)
        
        role = message.role
        data = message.model_dump()

        if role == "user":
            return HumanMessage(content=self._extract_content(data.get("content")))
        elif role == "system":
            return SystemMessage(content=self._extract_content(data.get("content")))
        elif role == "assistant":
            content = self._extract_content(data.get("content"))

            lc_tool_calls = []
            for tc in data.get("tool_calls", []):
                func = tc.get("function", {})
                raw_args = tc.get("arguments", "{}")
                lc_tool_calls.append(ToolCall(
                    id = tc.get("id", ""),
                    name = func.get("name", ""),
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                ))
            return AIMessage(content=content, tool_calls=lc_tool_calls)
        elif role == "tool":
            return ToolMessage(
                content=self._extract_content(data.get("content")),
                tool_call_id=data.get("tool_call_id", "")
            )

        raise ValueError(f"Unsupported message role: {role!r}")
    
    def preprocess(self, response_agent_request: ResponsesAgentRequest):
        if isinstance(response_agent_request, dict):
            response_agent_request = ResponsesAgentRequest(**response_agent_request)
        
        request = {
            **response_agent_request.model_dump(),
            "messages":[self.convert_message(m) for m in response_agent_request.input]
        }

        return request
    
    @staticmethod
    def _extract_text(content) -> str:
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(p for p in parts if p)
        return str(content)
    
    def _messages_to_output_items(self, messages: list) -> list[dict]:
        output: list[dict] = []

        for message in messages:
            if isinstance(message, (HumanMessage, SystemMessage)):
                continue

            elif isinstance(message, AIMessage):
                text = self._extract_text(message.content)
                if text:
                    output.append({
                        "type": "message",
                        "id": getattr(message, "id", None) or str(uuid.uuid4()),
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]
                    })
                
                for tc in message.tool_calls or []:
                    raw_args = tc.get("args", {})
                    output.append({
                        "type": "function_call",
                        "id": tc.get("id", "") or str(uuid.uuid4()),
                        "call_id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(raw_args) if isinstance(raw_args, dict) else raw_args,
                    })
            
            elif isinstance(message, ToolMessage):
                output.append({
                    "type": "function_call_output",
                    "id": getattr(message, "id", None) or str(uuid.uuid4()),
                    "call_id": message.tool_call_id,
                    "output": message.content
                })
        
        return output
    
    def postprocess(self, result: dict) -> ResponsesAgentResponse:
        messages = result.get("messages", [])
        output = self._messages_to_output_items(messages)
        return ResponsesAgentResponse(output=output)
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        if isinstance(request, dict):
            request = ResponsesAgentRequest(**request)
        
        chat_completion_request = self.preprocess(request)
        result = self.agent.invoke(chat_completion_request)
        return self.postprocess(result)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        if isinstance(request, dict):
            request = ResponsesAgentRequest(**request)

        chat_completion_request = self.preprocess(request)

        for chunk in self.agent.stream(chat_completion_request):
            step_messages: list = []
            if "agent" in chunk:
                step_messages = chunk["agent"].get("messages", [])
            elif "model" in chunk:
                step_messages = chunk["model"].get("messages", [])
            elif "tools" in chunk:
                step_messages = chunk["tools"].get("messages", [])
            
            for message in step_messages:
                if isinstance(message, (HumanMessage, SystemMessage)):
                    continue

                elif isinstance(message, AIMessage):
                    text = self._extract_text(message.content)
                    if text:
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_text_output_item(
                                text=text,
                                id=getattr(message, "id", None) or str(uuid.uuid4())
                            )
                        )
                    
                    for tc in message.tool_calls or []:
                        raw_args = tc.get("args", {})
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_function_call_item(
                                id=tc.get("id", ""),
                                call_id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=json.dumps(raw_args) if isinstance(raw_args, dict) else raw_args
                            )
                        )
                
                elif isinstance(message, ToolMessage):
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_function_call_output_item(
                            call_id=message.tool_call_id,
                            output=message.content
                        )
                    )


if __name__ == "__main__":
    # Test
    chatbot = LangChainAgentModel()
    chatbot.load_context(None)

    
    for i in chatbot.predict_stream(request = ResponsesAgentRequest(input=[Message(role="user", content="Qui Horus Lupercal a trahi ?")])):
        print("\n-----\n")

        if i.item["type"] == "function_call":
            print(f"Called {i.item['name']} with arguments: {i.item['arguments']}")

        elif i.item["type"] == "function_call_output":
            print("Tool call result:\n", i.item["output"])

        elif i.item["type"] == "message":
            print("AI message:\n", i.item["content"][0]["text"])

        else:
            raise ValueError("Unrecognized output.")
