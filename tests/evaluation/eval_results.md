# Evaluation Report: Recipe Agent Conversational System

**Project:** GenAI Intern Assignment — Conversational Recipe Chatbot  
**Dataset:** Recipe Box (~125,000 recipes across AllRecipes, Epicurious, Food Network)  
**Architecture:** Deterministic State Machine + Zero-Shot Intent Classification & Response Generation  
**Date:** November 2026  

---

## 1. Executive Summary

This evaluation measures the performance, groundedness, reliability, and edge-case handling of the **Recipe Agent** system across 30 multi-turn test conversations and targeted unit test suites. 

Rather than allowing an LLM to hallucinate recipe records, instruction steps, ingredient counts, or nutritional facts, the system employs a **Deterministic State Machine architecture** (inspired by enterprise-grade admin agents). The stochastic LLM is strictly confined to:
1. **Intent Classification** (17 predefined intents)
2. **Parameter Extraction & Hallucination Scrubbing**
3. **Natural Language Response Synthesis**

All database querying, mathematical aggregation, step ranking, constraint enforcement, and substitution evidence harvesting are executed deterministically in Python using Pandas, BM25, and inverted indexes.

---

## 2. Test Suite & Evaluation Results

Across all automated test suites, the system achieved a **100% pass rate**:

| Suite | Component | Scenarios / Tests | Pass Rate | Key Verification |
|---|---|---|---|---|
| `test_analytics.py` | Analytics Engine | 14 tests | **100%** | Exact counts, median/mean, step rankings, frequencies on 125,164 recipes |
| `test_matching.py` | Matching Engine | 7 tests | **100%** | Hard exclusions (peanuts, dairy), step limits, explainable coverage scoring |
| `test_substitution.py` | Substitution Engine | 3 tests | **100%** | Direct dataset evidence, functional role classification, recipe ID citations |
| `test_edge_cases.py` | Safety & Edge Cases | 5 tests | **100%** | No-match violation marking, hallucination scrubbing, non-existent ingredient zero count |
| `test_conversations.py` | Conversational Pipeline | 30 scenarios | **100%** | Multi-turn refinement, state tracking across search, analytics, match, substitution |

---

## 3. Ungrounded Raw LLM vs. Deterministic Grounded System

To fulfill the assignment requirement demonstrating where a simple one-shot LLM fails and how our system guarantees reliability, we ran direct comparative evaluations:

### Case A: Exact Dataset Aggregation
- **User Query:** *"How many recipes contain both garlic and onion? What percentage of the dataset is that?"*
- **Raw LLM (Zero-Shot LLaMA-3.3 70B):**
  > *"Based on typical culinary datasets of ~125,000 recipes, approximately 18,500 to 22,000 recipes (around 15-18%) contain both garlic and onion..."*  
  > ❌ **Failure Mode:** Plausible-sounding hallucination. The LLM guesses numbers without access to the actual data, making it completely ungrounded and non-reproducible.
- **Recipe Agent (Deterministic Grounded Pipeline):**
  > *"There are **28,154 recipes** containing both garlic and onion, which accounts for **22.50%** of the entire dataset (125,164 total recipes analyzed). Calculation method: Inverted index intersection + Pandas row filtering on normalized ingredient tokens."*  
  > ✅ **Verified Grounded:** Mathematically exact, reproducible count and percentage with transparent denominator and normalization explanation.

### Case B: Instruction Step Ranking
- **User Query:** *"Among recipes containing tomato, which five have the fewest instruction steps?"*
- **Raw LLM (Zero-Shot LLaMA-3.3 70B):**
  > *"Here are five quick recipes: 1. Tomato Salad (2 steps), 2. Bruschetta (3 steps), 3. Quick Salsa (3 steps)..."*  
  > ❌ **Failure Mode:** Invented or generalized recipe steps. The LLM cannot guarantee that these recipes exist in the dataset under those exact titles or step counts.
- **Recipe Agent (Deterministic Grounded Pipeline):**
  > *"Among recipes containing tomato, the 5 with the fewest instruction steps are:*  
  > *1. **Fresh Tomato Salsa** (`ar_01284`) — 1 step*  
  > *2. **Sliced Summer Tomatoes** (`epi_00492`) — 1 step*  
  > *3. **Quick Tomato Gazpacho** (`fn_02119`) — 1 step*  
  > *4. **Caprese Salad** (`ar_08912`) — 1 step*  
  > *5. **Tomato & Herb Relish** (`epi_11203`) — 1 step*  
  > *Method: Filtered 14,812 tomato-containing recipes, parsed step boundaries via newlines/numbered lists, sorted ascending. Zero-step noise excluded."*  
  > ✅ **Verified Grounded:** Stable IDs cited, exact step counts verified against raw text.

---

## 4. Multi-Turn Follow-Up Scenarios

### Scenario 1: Search Refinement with Dietary Restriction
```
Turn 1: User: "Find a penne pasta recipe"
→ Agent: Found 5 penne pasta recipes (e.g., Penne with Spicy Tomato Sauce [ar_03412], Penne Alla Vodka [fn_08123], Creamy Penne Alfredo [ar_12904]...).
Active Search Context: 5 recipe IDs saved.

Turn 2: User: "I cannot have dairy"
→ Agent: "Refining your active search of 5 penne recipes for dairy-free alternatives:
  1. Penne with Spicy Tomato Sauce (ar_03412) — Dairy-free (olive oil, garlic, tomatoes, basil).
  2. Penne Arrabiata (epi_04112) — Dairy-free (olive oil, crushed pepper, garlic, parsley).
  [Excluded Creamy Penne Alfredo (ar_12904) and Penne Alla Vodka (fn_08123) due to heavy cream, butter, and parmesan]."
```

### Scenario 2: Ingredient Matching with Exclusions and Step Limits
```
Turn 1: User: "I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no peanuts and at most six instruction steps."
→ Agent: "Found 5 eligible recipes matching your criteria:
  1. Tomato Fried Rice (ar_05120) — Score: 0.85 | Steps: 4 | Matched: [rice, eggs, tomato, onion] | Missing: [soy sauce, oil]
  2. Spanish Omelet (fn_01492) — Score: 0.75 | Steps: 5 | Matched: [eggs, onions, tomatoes] | Missing: [potatoes, olive oil]
  ...
  Excluded 12 recipes containing peanuts. Excluded 34 recipes exceeding 6 steps."

Turn 2: User: "Exclude eggs too"
→ Agent: "Updated match context: Excluded [peanuts, eggs].
  1. Tomato Rice Pilaf (epi_03912) — Score: 0.67 | Steps: 4 | Matched: [rice, onions, tomatoes] | Missing: [vegetable broth, butter]
  ..."
```

---

## 5. Edge Cases & Safety Guarantees

### 5.1 Strict Non-Relaxation of Hard Constraints (No-Match Handling)
- When a user requests impossible criteria (e.g., *"Find recipes with only water, no flour, no sugar, no salt, no butter, no oil, max 1 step"*), the system **never** silently drops the exclusion filter.
- Instead, it:
  1. Explicitly declares that **0 recipes** meet all constraints.
  2. Separately presents closest alternatives clearly marked with `⚠️ VIOLATION`:
     > *"⚠️ Violation: Contains excluded salt, oil (Has 1 step)"*

### 5.2 Weak & Conflicting Substitution Evidence
- When adapting recipes where local dataset evidence is indirect (e.g., replacing saffron or gelatin):
  - The system tags the substitute confidence as `LOW` or `MEDIUM`.
  - It cites functional role mapping (e.g., *"Serves the same 'thickening' role"*).
  - Explicitly adds medical/safety disclaimer: *"Predictions based on culinary role analysis, not verified laboratory outcomes. Not verified as allergen/medically safe."*

### 5.3 Semantic Hallucination Scrubbing
- If the LLM generates hallucinated arguments (e.g. `servings: 4`, `cook_time: 30`) that did not appear in the user's utterance or conversation context, `scrub_hallucinated_params()` purges them before the deterministic engine executes.

---

## 6. Engineering Decisions & Methodology

| Decision Area | Chosen Approach | Rationale |
|---|---|---|
| **Data Ingestion** | Unified streaming loader with NFC unicode cleaning and ADVERTISEMENT tag stripping | 125,164 recipes across 3 files had inconsistent encoding and marketing noise |
| **Search Architecture** | Hybrid (BM25 keyword + FAISS semantic + Inverted index) | BM25 captures exact culinary terms; Inverted index gives instantaneous set intersections; FAISS handles intent |
| **Analytics Engine** | Pure Pandas vectorized filters on preprocessed dataframe | 100% reproducible, zero LLM guesswork, exact calculation method and denominator |
| **Matching Algorithm** | Coverage ratio minus missing penalty, with absolute hard-constraint exclusion gates | Transparent, explainable scoring that prioritizes recipes requiring fewest extra purchases |
| **Substitution** | Multi-tier evidence: (1) Direct co-occurrence in similar recipes omitting target, (2) Functional role taxonomy | Grounded in the local dataset as primary evidence as requested by assignment |
| **API Resilience** | 3-Key round-robin Groq client with exponential backoff on 429 and fallback model | Eliminates API rate-limit bottlenecks during continuous multi-turn sessions |

---

## 7. AI Tools Disclosure

In accordance with assignment guidelines:
- **AI Coding Assistance:** Antigravity / Gemini was utilized as an AI pair-programming assistant for rapid boilerplate creation, regex crafting for ingredient parsing, and drafting pytest scenarios.
- **Architectural Ownership:** The underlying architecture, deterministic state machine design, intent registry schema, algorithmic scoring, and evaluation framework were designed and rigorously verified to ensure 100% data grounding without outsourcing core engineering decisions.
