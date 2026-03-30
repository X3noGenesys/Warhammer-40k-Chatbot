import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from src.rag.agent import LangChainAgentModel
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.responses_helpers import Message


@dataclass
class ToolCallExpectation:
    """Décrit un appel d'outil attendu."""
    tool_name: str

@dataclass
class EvalCase:
    """Un cas de test complet."""
    id: str
    question: str
    expected_tool_calls: list[ToolCallExpectation]
    expected_keywords: list[str]
    forbidden_keywords: list[str] = field(default_factory=list)
    description: str = ""


TEST_CASES: list[EvalCase] = [

    EvalCase(
        id="TC-01",
        description="Membres de la faction Space Marines Ultramarines",
        question="Quels sont les membres des Ultramarines ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["ultramarine", "space marine", "chapitre"],
    ),

    EvalCase(
        id="TC-02",
        description="Ennemis de Horus Lupercal",
        question="Qui Horus Lupercal a trahi ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["horus", "trahison", "imperium"],
    ),

    EvalCase(
        id="TC-03",
        description="Lieu d'une bataille (tookPlaceAt)",
        question="Où s'est déroulée la bataille de Calth ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["calth", "bataille", "word bearers"],
    ),

    EvalCase(
        id="TC-04",
        description="Arme manié par un personnage (wields)",
        question="Quelle arme Abaddon le Fléau manie-t-il ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["abaddon", "drach'nyen", "talon"],
    ),

    EvalCase(
        id="TC-05",
        description="Faction d'appartenance (memberOf)",
        question="À quelle faction appartient Eldrad Ulthran ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["eldar", "ulthwé", "eldrad"],
    ),

    EvalCase(
        id="TC-06",
        description="Divinité vénérée (worships)",
        question="Quel dieu les Thousand Sons vénèrent-ils ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["tzeentch", "chaos", "thousand sons"],
    ),

    EvalCase(
        id="TC-07",
        description="Entité dirigeante (leads)",
        question="Qui dirige les Black Legion ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["abaddon", "black legion", "chaos"],
    ),

    EvalCase(
        id="TC-08",
        description="Origine d'une entité (originatesFrom)",
        question="D'où vient Roboute Guilliman ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["macragge", "ultramar", "guilliman"],
    ),

    EvalCase(
        id="TC-09",
        description="Question hors-lore (refus attendu, pas d'outil)",
        question="Quel est le prix d'un café à Paris aujourd'hui ?",
        expected_tool_calls=[],   # Aucun outil attendu
        expected_keywords=["41ème millénaire", "calibré", "warhammer"],
        forbidden_keywords=["get_entity_id", "predict_relation_tail"],
    ),


    EvalCase(
        id="TC-10",
        description="Résolution floue du nom d'entité",
        question="Quels sont les alliés de l'Empereur de l'Humanitée ?",
        expected_tool_calls=[
            ToolCallExpectation(tool_name="get_entity_id"),
            ToolCallExpectation(tool_name="predict_relation_tail"),
        ],
        expected_keywords=["empereur", "imperium"],
    ),
]


@dataclass
class EvalResult:
    case_id: str
    description: str
    question: str
    passed: bool
    tool_call_score: float          
    keyword_score: float            
    forbidden_keyword_score: float 
    tool_call_details: list[dict] = field(default_factory=list)
    keyword_details: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    raw_tool_calls: list[dict] = field(default_factory=list)
    raw_response: str = ""


def _normalize(text: str) -> str:
    return text.lower().strip()


def _collect_stream_output(agent: LangChainAgentModel, question: str) -> tuple[list[dict], str]:
    """Exécute l'agent en mode stream et collecte tool calls + réponse finale."""
    tool_calls_seen: list[dict] = []
    final_text_parts: list[str] = []

    request = ResponsesAgentRequest(
        input=[Message(role="user", content=question)]
    )

    for event in agent.predict_stream(request):
        item = event.item
        if item["type"] == "function_call":
            tool_calls_seen.append({
                "name": item.get("name", ""),
                "arguments": json.loads(item.get("arguments", "{}"))
                             if isinstance(item.get("arguments"), str)
                             else item.get("arguments", {}),
            })
        elif item["type"] == "message":
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") == "output_text":
                    final_text_parts.append(block.get("text", ""))

    return tool_calls_seen, "\n".join(final_text_parts)


def evaluate_tool_calls(
    observed: list[dict],
    expected: list[ToolCallExpectation],
) -> tuple[float, list[dict]]:
    if not expected:
        score = 1.0 if not observed else 0.0
        details = [{"expected": "no tool calls",
                    "found": bool(observed),
                    "passed": not bool(observed)}]
        return score, details

    details = []
    matched_count = 0
    obs_idx = 0

    for exp in expected:
        found = False
        for i in range(obs_idx, len(observed)):
            if observed[i]["name"] == exp.tool_name:
                found = True
                obs_idx = i + 1
                matched_count += 1
                break

        details.append({
            "expected_tool": exp.tool_name,
            "passed": found,
        })

    score = matched_count / len(expected)
    return score, details


def evaluate_keywords(
    response: str,
    expected_kw: list[str],
    forbidden_kw: list[str],
) -> tuple[float, float, list[dict]]:
    resp_lower = response.lower()
    details = []

    kw_hits = 0
    for kw in expected_kw:
        found = kw.lower() in resp_lower
        if found:
            kw_hits += 1
        details.append({"keyword": kw, "expected": True, "found": found})

    kw_score = kw_hits / len(expected_kw) if expected_kw else 1.0

    fk_misses = 0
    for kw in forbidden_kw:
        found = kw.lower() in resp_lower
        if not found:
            fk_misses += 1
        details.append({"keyword": kw, "expected": False, "found": found})

    fk_score = fk_misses / len(forbidden_kw) if forbidden_kw else 1.0

    return kw_score, fk_score, details


def run_eval_case(agent: LangChainAgentModel, case: EvalCase, verbose: bool = False) -> EvalResult:
    if verbose:
        print(f"\n{'─'*60}")
        print(f"[{case.id}] {case.description}")
        print(f"  Q: {case.question}")

    errors: list[str] = []
    t0 = time.perf_counter()

    try:
        raw_tool_calls, raw_response = _collect_stream_output(agent, case.question)
    except Exception as exc:
        errors.append(f"Agent crash: {exc}")
        raw_tool_calls, raw_response = [], ""

    latency = time.perf_counter() - t0

    tc_score, tc_details = evaluate_tool_calls(raw_tool_calls, case.expected_tool_calls)
    kw_score, fk_score, kw_details = evaluate_keywords(
        raw_response, case.expected_keywords, case.forbidden_keywords
    )

    TOOL_THRESHOLD = 1.0
    KW_THRESHOLD = 0.6
    FK_THRESHOLD = 1.0

    passed = (
        tc_score >= TOOL_THRESHOLD
        and kw_score >= KW_THRESHOLD
        and fk_score >= FK_THRESHOLD
        and not errors
    )

    if verbose:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | tools={tc_score:.0%} kw={kw_score:.0%} fk={fk_score:.0%} t={latency:.1f}s")
        for d in tc_details:
            mark = "✓" if d["passed"] else "✗"
            print(f"    [{mark}] tool:{d['expected_tool']}")
        for d in kw_details:
            mark = "✓" if (d["expected"] == d["found"]) else "✗"
            print(f"    [{mark}] kw:'{d['keyword']}' found={d['found']}")
        if errors:
            for e in errors:
                print(f"    ⚠  {e}")

    return EvalResult(
        case_id=case.id,
        description=case.description,
        question=case.question,
        passed=passed,
        tool_call_score=tc_score,
        keyword_score=kw_score,
        forbidden_keyword_score=fk_score,
        tool_call_details=tc_details,
        keyword_details=kw_details,
        errors=errors,
        latency_s=round(latency, 2),
        raw_tool_calls=raw_tool_calls,
        raw_response=raw_response,
    )


def print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    passed = sum(r.passed for r in results)
    avg_tc = sum(r.tool_call_score for r in results) / total
    avg_kw = sum(r.keyword_score for r in results) / total
    avg_lat = sum(r.latency_s for r in results) / total

    sep = "═" * 60
    print(f"\n{sep}")
    print("RAPPORT D'ÉVALUATION")
    print(sep)
    print(f"  Cas testés      : {total}")
    print(f"  Réussis         : {passed}/{total}  ({passed/total:.0%})")
    print(f"  Score outil moy : {avg_tc:.0%}")
    print(f"  Score mots-clés : {avg_kw:.0%}")
    print(f"  Latence moyenne : {avg_lat:.1f}s")
    print(sep)

    header = f"  {'ID':<8} {'PASS':<6} {'TOOLS':<8} {'KW':<8} {'LAT':<8} DESCRIPTION"
    print(header)
    print("  " + "─" * 58)
    for r in results:
        status = "✅" if r.passed else "❌"
        print(
            f"  {r.case_id:<8} {status:<6} "
            f"{r.tool_call_score:<8.0%} {r.keyword_score:<8.0%} "
            f"{r.latency_s:<8.1f} {r.description[:35]}"
        )
    print(sep)

    failed = [r for r in results if not r.passed]
    if failed:
        print("\n  ── CAS ÉCHOUÉS — DÉTAIL ──")
        for r in failed:
            print(f"\n  [{r.case_id}] {r.description}")
            print(f"   Q: {r.question}")
            for d in r.tool_call_details:
                if not d["passed"]:
                    print(f"   ✗ Outil manquant : {d['expected_tool']}")
            for d in r.keyword_details:
                if d["expected"] != d["found"]:
                    label = "attendu absent" if d["expected"] else "interdit présent"
                    print(f"   ✗ Mot-clé '{d['keyword']}' → {label}")
            for e in r.errors:
                print(f"   ⚠  {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Évaluation de COGITATOR-SIGMA-VII")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Affiche les détails de chaque test au fil de l'exécution")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Chemin vers un fichier JSON pour sauvegarder les résultats")
    parser.add_argument("--case", "-c", type=str, default=None,
                        help="Exécuter uniquement un cas précis (ex: TC-02)")
    args = parser.parse_args()

    print("Initialisation de l'agent…")
    agent = LangChainAgentModel()
    agent.load_context(None)

    cases = TEST_CASES
    if args.case:
        cases = [c for c in TEST_CASES if c.id == args.case]
        if not cases:
            print(f"Aucun cas trouvé avec l'ID '{args.case}'.")
            return

    results: list[EvalResult] = []
    for case in cases:
        result = run_eval_case(agent, case, verbose=args.verbose)
        results.append(result)

    print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"Résultats sauvegardés → {output_path}")


if __name__ == "__main__":
    main()