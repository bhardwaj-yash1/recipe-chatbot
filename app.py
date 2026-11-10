"""
Recipe Agent — Streamlit Conversational Interface

A production-grade conversational recipe assistant powered by a
Deterministic State Machine architecture.  The LLM classifies intent
and formats gourmet culinary responses; all data queries, calculations,
constraint checks, and substitutions are executed deterministically.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# ── Ensure project root is on sys.path ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator, get_source_name
from config.settings import settings
from core.data_loader import load_all_recipes, load_processed_recipes, save_processed_recipes
from core.index_builder import build_all_indices
from core.models import Recipe
from engine.analytics_engine import AnalyticsEngine, AnalyticsResult
from engine.matching_engine import MatchingEngine, MatchResult
from engine.search_engine import SearchEngine
from engine.substitution_engine import SubstitutionEngine, SubstitutionResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gourmet AI Recipe Agent",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
# Startup: load data & build shared engines (cached across all users)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Warming up recipe intelligence & mounting 125,000 recipes…")
def get_shared_engines():
    """Load or build all data, indexes, and engines. Runs once and is cached."""
    processed_path = settings.processed_data_dir / "recipes.jsonl"

    if processed_path.exists():
        recipes = load_processed_recipes(processed_path)
    else:
        recipes = load_all_recipes()
        save_processed_recipes(recipes)
        build_all_indices(recipes)

    faiss_path = settings.processed_data_dir / "faiss_index.bin"
    if not faiss_path.exists():
        build_all_indices(recipes)

    search_engine = SearchEngine(recipes, settings.processed_data_dir)
    search_engine.load_all_indexes()
    analytics_engine = AnalyticsEngine(recipes)
    matching_engine = MatchingEngine(recipes)
    substitution_engine = SubstitutionEngine(recipes, inverted_index=search_engine._inverted)
    llm_client = LLMClient()

    # Source breakdown
    sources: Dict[str, int] = {}
    for r in recipes:
        src = get_source_name(r.source)
        sources[src] = sources.get(src, 0) + 1

    return {
        "search": search_engine,
        "analytics": analytics_engine,
        "matching": matching_engine,
        "substitution": substitution_engine,
        "llm": llm_client,
        "recipes": recipes,
        "total_recipes": len(recipes),
        "sources": sources,
    }


engines = get_shared_engines()
total_recipes = engines["total_recipes"]
sources = engines["sources"]

# Per-session orchestrator (stores conversational context & multi-turn memory)
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator(
        search_engine=engines["search"],
        analytics_engine=engines["analytics"],
        matching_engine=engines["matching"],
        substitution_engine=engines["substitution"],
        llm_client=engines["llm"],
    )
orchestrator: Orchestrator = st.session_state.orchestrator


# ─────────────────────────────────────────────────────────────────────
# Sidebar: System Status & Quick Navigation
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍳 Gourmet Recipe Agent")
    st.caption("AI-Powered Culinary Assistant")
    st.markdown("---")

    # Metrics
    st.markdown("### 📊 Dataset Overview")
    st.metric("Total Recipes", f"{total_recipes:,}")
    c1, c2, c3 = st.columns(3)
    c1.caption(f"**Food Network**\n\n{sources.get('Food Network', 0):,}")
    c2.caption(f"**AllRecipes**\n\n{sources.get('AllRecipes', 0):,}")
    c3.caption(f"**Epicurious**\n\n{sources.get('Epicurious', 0):,}")

    st.markdown("---")
    st.markdown("### ⚙️ Session Status")
    st.caption("🟢 **Engines**: Online (FAISS + BM25 + Pandas)")
    st.caption(f"🤖 **LLM Calls**: {orchestrator.llm.call_count}")

    # Active filters from context
    active_restr = orchestrator.context.filters.get("restrictions", [])
    if active_restr:
        st.info(f"Active Filters: {', '.join(active_restr)}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", width="stretch"):
        st.session_state.messages = []
        orchestrator.reset()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    with st.container(border=True):
        st.markdown("""
        **Deterministic State Machine**
        - **LLM**: Intent classification & culinary formatting
        - **Search**: BM25 sparse + FAISS dense semantic hybrid
        - **Analytics**: Vectorized Pandas (0% math hallucination)
        - **Matching**: Strict non-relaxation constraint engine
        - **Substitutions**: Grounded in 125k local dataset recipes
        """)


# ─────────────────────────────────────────────────────────────────────
# Main Chat Header
# ─────────────────────────────────────────────────────────────────────
st.title("🍳 Recipe Agent")
st.markdown(
    f"Search across **{total_recipes:,} recipes**, compute exact corpus statistics, "
    "match pantry ingredients with strict constraints, and discover evidence-backed substitutions."
)
st.markdown("---")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────────
# Welcome Screen (shown when no messages exist)
# ─────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### Welcome! How can I assist your cooking today?")
    st.caption("Click any example prompt below or type your own question in the chat bar.")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 🔍 Recipe Search & Cooking")
            st.write("Full ingredients with exact quantities and complete step-by-step directions.")
            if st.button("🍝 Give me a recipe for red sauce pasta", width="stretch"):
                st.session_state["pending_example"] = "give me recipe for red sauce pasta"
                st.rerun()
            if st.button("💡 What should I cook today?", width="stretch"):
                st.session_state["pending_example"] = "what should i cook today"
                st.rerun()

        with col2:
            with st.container(border=True):
                st.markdown("#### 📊 Exact Corpus Analytics")
                st.write("Zero numerical hallucinations. Calculated directly over 125,000 recipes.")
                if st.button("🧄 How many recipes contain both garlic and onion?", width="stretch"):
                    st.session_state["pending_example"] = "How many recipes contain both garlic and onion? What percentage of the dataset is that?"
                    st.rerun()
                if st.button("⏱️ 5 recipes with fewest steps among tomato dishes", width="stretch"):
                    st.session_state["pending_example"] = "Among recipes containing tomato, which five have the fewest instruction steps?"
                    st.rerun()

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.markdown("#### 🧑‍🍳 Smart Pantry Matcher")
            st.write("Rank recipes by available ingredients with non-relaxable hard dietary constraints.")
            if st.button("🥗 I have eggs, onions, tomatoes, and rice", width="stretch"):
                st.session_state["pending_example"] = "I have eggs, onions, tomatoes, and rice. Suggest recipes with no peanuts and at most 6 steps."
                st.rerun()

    with col4:
        with st.container(border=True):
            st.markdown("#### 🔄 Evidence-Based Substitutions")
            st.write("Substitutions supported by co-occurrence evidence in the local dataset.")
            if st.button("🍌 Replace eggs in banana bread", width="stretch"):
                st.session_state["pending_example"] = "Take this banana bread recipe and remove eggs. Use only ingredients that appear in the dataset."
                st.rerun()

    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────
# Helper: Extract Technical Execution Metadata for Technical Drawer
# ─────────────────────────────────────────────────────────────────────
def extract_turn_metadata(orchestrator: Orchestrator, duration: float) -> Dict[str, Any]:
    """Capture execution details from the orchestrator for technical review."""
    meta: Dict[str, Any] = {
        "duration_seconds": round(duration, 2),
        "intent": "N/A",
        "engine": "Deterministic Engine",
        "params": {},
        "provenance": [],
    }
    if not orchestrator.context.turns:
        return meta

    last_turn = orchestrator.context.turns[-1]
    meta["intent"] = last_turn.intent.value
    meta["params"] = last_turn.params

    data = last_turn.result_data
    if isinstance(data, list) and data:
        if isinstance(data[0], Recipe):
            meta["engine"] = "SearchEngine (FAISS Dense + BM25 Sparse Hybrid)"
            meta["provenance"] = [
                {"title": r.title, "source": get_source_name(r.source), "dataset_id": r.recipe_id, "steps": len(r.instruction_steps)}
                for r in data[:5]
            ]
        elif isinstance(data[0], MatchResult):
            meta["engine"] = "MatchingEngine (Strict Constraint Non-Relaxation)"
            meta["provenance"] = [
                {
                    "title": m.recipe.title,
                    "source": get_source_name(m.recipe.source),
                    "dataset_id": m.recipe.recipe_id,
                    "score": round(m.score, 3),
                    "matched": len(m.matched_ingredients),
                    "missing": len(m.missing_ingredients),
                }
                for m in data[:5]
            ]
        elif isinstance(data[0], SubstitutionResult):
            meta["engine"] = "SubstitutionEngine (Local Dataset Co-Occurrence & Roles)"
            meta["provenance"] = [
                {
                    "ingredient_replaced": s.ingredient_replaced,
                    "roles": s.ingredient_roles,
                    "candidates": [c.substitute_name for c in s.candidates[:3]],
                }
                for s in data
            ]
    elif isinstance(data, AnalyticsResult):
        meta["engine"] = "AnalyticsEngine (Vectorized Pandas Aggregations)"
        meta["provenance"] = [{
            "metric": data.metric,
            "calculated_value": data.value,
            "recipes_analyzed": data.total_analyzed,
            "total_dataset": data.total_dataset,
            "method": data.method,
        }]

    return meta


# ─────────────────────────────────────────────────────────────────────
# Display Existing Messages
# ─────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🧑‍💻" if role == "user" else "🍳"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # Technical Metadata Expander (Interview/Evaluation Drawer)
        meta = msg.get("meta")
        if meta and role == "assistant":
            with st.expander("🔬 Technical Execution Details & Dataset Provenance", expanded=False):
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Classified Intent", meta.get("intent", "N/A"))
                m_col2.metric("Deterministic Engine", meta.get("engine", "N/A").split(" ")[0])
                m_col3.metric("Latency", f"{meta.get('duration_seconds', 0)}s")

                st.caption(f"**Full Engine**: {meta.get('engine', 'N/A')}")
                if meta.get("params"):
                    st.caption(f"**Extracted Parameters**: `{json.dumps(meta.get('params'))}`")

                if meta.get("provenance"):
                    st.markdown("**Dataset Provenance & Stable IDs (Preserved for Evaluation):**")
                    st.json(meta["provenance"])


# ─────────────────────────────────────────────────────────────────────
# Handle Input
# ─────────────────────────────────────────────────────────────────────
user_input = None
if "pending_example" in st.session_state:
    user_input = st.session_state.pop("pending_example")
else:
    user_input = st.chat_input("Ask a recipe question, search for dishes, match pantry items, or compute statistics…")

if user_input:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # 2. Process message through orchestrator
    with st.chat_message("assistant", avatar="🍳"):
        t0 = time.time()
        with st.spinner("Preparing culinary recommendations…"):
            try:
                response = orchestrator.handle_message(user_input)
            except Exception as e:
                logger.exception("Error processing message")
                response = f"⚠️ An error occurred while processing: {str(e)}\n\nPlease try rephrasing your request."
        duration = time.time() - t0

        st.markdown(response)

        # Technical Drawer
        meta = extract_turn_metadata(orchestrator, duration)
        with st.expander("🔬 Technical Execution Details & Dataset Provenance", expanded=False):
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Classified Intent", meta.get("intent", "N/A"))
            m_col2.metric("Deterministic Engine", meta.get("engine", "N/A").split(" ")[0])
            m_col3.metric("Latency", f"{meta.get('duration_seconds', 0)}s")

            st.caption(f"**Full Engine**: {meta.get('engine', 'N/A')}")
            if meta.get("params"):
                st.caption(f"**Extracted Parameters**: `{json.dumps(meta.get('params'))}`")

            if meta.get("provenance"):
                st.markdown("**Dataset Provenance & Stable IDs (Preserved for Evaluation):**")
                st.json(meta["provenance"])

    # 3. Store in session
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "meta": meta,
    })

    # Update sidebar metrics
    st.rerun()
