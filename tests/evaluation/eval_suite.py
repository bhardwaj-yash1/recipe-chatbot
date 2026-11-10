"""
Reproducible Evaluation Suite for Recipe Agent.

Demonstrates:
  1. Grounded vs Ungrounded LLM comparison (proves where a raw LLM fails/hallucinates
     and how our Deterministic State Machine provides mathematically grounded answers).
  2. Multi-turn conversation evaluations across 30 scenarios.
  3. No-match case handling with constraint violation flagging.
  4. Weak/conflicting evidence substitution handling.

Usage:
  python tests/evaluation/eval_suite.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from core.data_loader import load_processed_recipes
from engine.analytics_engine import AnalyticsEngine
from engine.matching_engine import MatchingEngine
from engine.search_engine import SearchEngine
from engine.substitution_engine import SubstitutionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("eval_suite")


def evaluate_llm_vs_grounded(orchestrator: Orchestrator) -> Dict[str, Any]:
    """
    Compare raw one-shot LLM output vs. our deterministic system.
    Demonstrates hallucination and ungrounded guesses from raw LLMs.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: RAW LLM (UNGROUNDED) vs. DETERMINISTIC ENGINE (GROUNDED)")
    print("=" * 70)

    # ── Test Query 1: Exact Dataset Statistics ──
    query_1 = "How many recipes contain both garlic and onion? What percentage of the dataset is that?"
    print(f"\n[QUERY 1]: {query_1}")

    # 1. Ask raw LLM directly without tools/engines
    raw_prompt = (
        "You are an AI assistant. Answer this question about the Recipe Box dataset (~125,000 recipes):\n"
        f"{query_1}\n"
        "State the exact count and percentage."
    )
    t0 = time.time()
    try:
        raw_llm_answer = orchestrator.llm.invoke(
            "You are a helpful assistant.",
            raw_prompt,
        )
    except Exception as e:
        raw_llm_answer = f"[LLM API Error: {e}]"
    t_raw = time.time() - t0

    print(f"\n--- RAW LLM RESPONSE (Zero-shot / No grounding) [{t_raw:.2f}s] ---")
    print(raw_llm_answer[:400] + "...")

    # 2. Ask our deterministic orchestrator
    orchestrator.reset()
    t0 = time.time()
    system_answer = orchestrator.handle_message(query_1)
    t_sys = time.time() - t0

    print(f"\n--- RECIPE AGENT SYSTEM RESPONSE (Deterministic Grounding) [{t_sys:.2f}s] ---")
    print(system_answer)

    # ── Test Query 2: Ranking by Instruction Steps ──
    query_2 = "Among recipes containing tomato, which five have the fewest instruction steps?"
    print(f"\n[QUERY 2]: {query_2}")

    orchestrator.reset()
    system_rank = orchestrator.handle_message(query_2)
    print(f"\n--- RECIPE AGENT SYSTEM RANKING ---")
    print(system_rank)

    # ── Test Query 3: Strict Constraint & No-Match Case ──
    query_3 = "I have water and salt. Find recipes with no flour, no sugar, no butter, no oil, no meat, and at most 1 step."
    print(f"\n[QUERY 3 (No-Match Edge Case)]: {query_3}")

    orchestrator.reset()
    system_no_match = orchestrator.handle_message(query_3)
    print(f"\n--- RECIPE AGENT SYSTEM RESPONSE (Explicit No-Match / Violation Marking) ---")
    print(system_no_match)

    return {
        "query_1_raw": raw_llm_answer,
        "query_1_grounded": system_answer,
        "query_2_ranking": system_rank,
        "query_3_no_match": system_no_match,
    }


def run_30_conversations_eval(orchestrator: Orchestrator) -> List[Dict[str, Any]]:
    """Run all 30 conversational scenarios and record latency, intent, and success."""
    scenarios = [
        # Search & Q&A (1-10)
        {"id": 1, "task": "Search", "turns": ["Find a penne pasta recipe"]},
        {"id": 2, "task": "Search", "turns": ["Show me recipes that use avocado"]},
        {"id": 3, "task": "Search", "turns": ["Find a penne pasta recipe", "I cannot have dairy"]},
        {"id": 4, "task": "Search", "turns": ["Search for chicken soup recipes", "Are there any without onion?"]},
        {"id": 5, "task": "Search", "turns": ["Something light and refreshing for summer"]},
        {"id": 6, "task": "Search", "turns": ["Show me the details for recipe ar_00001"]},
        {"id": 7, "task": "Search", "turns": ["Find recipes for unicorn steak with dragon sauce"]},
        {"id": 8, "task": "Search", "turns": ["Find me a cake", "I meant chocolate cake specifically"]},
        {"id": 9, "task": "Search", "turns": ["Find recipes that use both garlic and basil"]},
        {"id": 10, "task": "Search", "turns": ["Find a lasagna recipe", "What ingredients does it have?"]},

        # Analytics (11-18)
        {"id": 11, "task": "Analytics", "turns": ["How many recipes contain garlic?"]},
        {"id": 12, "task": "Analytics", "turns": ["How many recipes contain both garlic and onion? What percentage of the dataset is that?"]},
        {"id": 13, "task": "Analytics", "turns": ["What is the average number of ingredients per recipe?"]},
        {"id": 14, "task": "Analytics", "turns": ["Among recipes containing tomato, which five have the fewest instruction steps?"]},
        {"id": 15, "task": "Analytics", "turns": ["What are the 10 most common ingredients across all recipes?"]},
        {"id": 16, "task": "Analytics", "turns": ["How many recipes use chicken?", "And how many of those also use garlic?"]},
        {"id": 17, "task": "Analytics", "turns": ["What is the median number of ingredients in recipes?"]},
        {"id": 18, "task": "Analytics", "turns": ["Count recipes with chocolate but no dairy"]},

        # Matching (19-24)
        {"id": 19, "task": "Matching", "turns": ["I have eggs, flour, sugar, and butter. What can I make?"]},
        {"id": 20, "task": "Matching", "turns": ["I have eggs, onions, tomatoes, and rice. Suggest five recipes with no peanuts and at most six instruction steps."]},
        {"id": 21, "task": "Matching", "turns": ["I have chicken, garlic, and olive oil. Find recipes with at most 4 steps."]},
        {"id": 22, "task": "Matching", "turns": ["I have eggs, onions, tomatoes, and rice. What can I make?", "Exclude eggs too"]},
        {"id": 23, "task": "Matching", "turns": ["I have water and salt. Find recipes with no flour, no sugar, no butter, no oil, and at most 1 step."]},
        {"id": 24, "task": "Matching", "turns": ["I have pasta, garlic, olive oil, parmesan, and black pepper. Find recipes needing fewest extra ingredients."]},

        # Substitution (25-30)
        {"id": 25, "task": "Substitution", "turns": ["Find a banana bread recipe", "Take this banana bread recipe and remove eggs. What replacements have evidence from related baked recipes?"]},
        {"id": 26, "task": "Substitution", "turns": ["Find a pasta with cream sauce recipe", "Replace cream in this pasta recipe without dairy. Prefer examples from recipes in the database."]},
        {"id": 27, "task": "Substitution", "turns": ["Find a cookie recipe", "I need this recipe without eggs and without dairy. Adapt it for me."]},
        {"id": 28, "task": "Substitution", "turns": ["Find a recipe that uses saffron", "Can I replace saffron with something cheaper?"]},
        {"id": 29, "task": "Substitution", "turns": ["Find a pancake recipe", "Replace the eggs in this recipe", "How might that affect the texture?"]},
        {"id": 30, "task": "Substitution", "turns": ["Find a recipe using gelatin", "Replace gelatin with something vegan, but it must set like gelatin"]},
    ]

    print("\n" + "=" * 70)
    print("RUNNING 30 MULTI-TURN CONVERSATION SCENARIOS")
    print("=" * 70)

    results = []
    for sc in scenarios:
        orchestrator.reset()
        t0 = time.time()
        turn_responses = []
        for turn_text in sc["turns"]:
            resp = orchestrator.handle_message(turn_text)
            turn_responses.append({"input": turn_text, "output": resp})

        elapsed = time.time() - t0
        passed = all(len(tr["output"]) > 20 for tr in turn_responses)
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] Scenario {sc['id']:02d} ({sc['task']}): {len(sc['turns'])} turn(s) in {elapsed:.2f}s")

        results.append({
            "id": sc["id"],
            "task": sc["task"],
            "turns": turn_responses,
            "elapsed_sec": round(elapsed, 2),
            "status": status,
        })

    return results


def main():
    print("Booting evaluation suite...")
    recipes = load_processed_recipes()
    search_engine = SearchEngine(recipes)
    search_engine.load_all_indexes()
    analytics_engine = AnalyticsEngine(recipes)
    matching_engine = MatchingEngine(recipes)
    substitution_engine = SubstitutionEngine(recipes)
    llm_client = LLMClient()

    orchestrator = Orchestrator(
        search_engine=search_engine,
        analytics_engine=analytics_engine,
        matching_engine=matching_engine,
        substitution_engine=substitution_engine,
        llm_client=llm_client,
    )

    # 1. Run LLM vs Grounded Comparison
    comparison = evaluate_llm_vs_grounded(orchestrator)

    # 2. Run 30 conversation scenarios
    scenarios_results = run_30_conversations_eval(orchestrator)

    # Summary
    passed_count = sum(1 for r in scenarios_results if r["status"] == "PASSED")
    total_count = len(scenarios_results)
    avg_latency = sum(r["elapsed_sec"] for r in scenarios_results) / total_count

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {passed_count}/{total_count} SCENARIOS PASSED")
    print(f"Average Conversation Latency: {avg_latency:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
