"""Memory Plus — enhanced memory Core for Tater.

A fork of the stock Memory Core (``cores/memory_core.py``, TaterTotterson/Tater_Shop)
with pseudo-semantic lexical retrieval, correctable fact updates, first-run
backfill, custom concept synonyms, LLM-proposed nightly consolidation, and
stock-core coexistence (shadow mode / import).

Single-file Core: stdlib + modules Tater already ships. No new pip dependencies.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from helpers import redis_client

__version__ = "1.0.0"
MIN_TATER_VERSION = "187"
CORE_DESCRIPTION = (
    "Enhanced memory: episodic recall, smart lexical retrieval, profile cards, nightly consolidation."
)
TAGS = ["memory", "personalization", "recall"]

SETTINGS_KEY = "memory_plus_core_settings"
RUNNING_KEY = "memory_plus_core_running"
USER_DOC_PREFIX = "memory_plus_core:user"
ROOM_DOC_PREFIX = "memory_plus_core:room"
CURSOR_PREFIX = "memory_plus_core:cursor"
BACKUP_PREFIX = "memory_plus_core:backup"
META_KEY = "memory_plus_core:meta"

logger = logging.getLogger("memory_plus_core")

CORE_SETTINGS = {
    "category": "Memory Plus Core Settings",
    "hydra_tools_require_running": False,
    "required": {
        "interval_seconds": {
            "label": "Interval (sec)",
            "type": "select",
            "default": "120",
            "options": [
                {"value": "60", "label": "1 minute"},
                {"value": "120", "label": "2 minutes"},
                {"value": "180", "label": "3 minutes"},
                {"value": "300", "label": "5 minutes"},
                {"value": "600", "label": "10 minutes"},
                {"value": "900", "label": "15 minutes"},
                {"value": "1800", "label": "30 minutes"},
                {"value": "3600", "label": "1 hour"},
            ],
            "description": "How often memory extraction runs.",
        },
        "lookback_limit": {
            "label": "Lookback Limit",
            "type": "number",
            "default": 120,
            "description": "Max new messages processed per scope per pass (10-500).",
        },
        "initial_backfill_messages": {
            "label": "Initial Backfill Messages",
            "type": "number",
            "default": 500,
            "description": (
                "On first encounter of a scope, process up to this many historical messages "
                "in lookback-sized batches across cycles."
            ),
        },
        "min_confidence": {
            "label": "Min Confidence",
            "type": "number",
            "default": 0.55,
            "description": "Minimum confidence required before storing a fact.",
        },
        "inject_context": {
            "label": "Inject Memory Context",
            "type": "checkbox",
            "default": True,
            "description": (
                "Inject memory into the prompt. Turn off for shadow mode: extract and store, "
                "but let the stock Memory Core inject (A/B testing)."
            ),
        },
        "max_facts": {
            "label": "Max Injected Facts",
            "type": "number",
            "default": 30,
            "description": "Facts injected per scope.",
        },
        "max_episodes": {
            "label": "Max Injected Episodes",
            "type": "number",
            "default": 5,
            "description": "Episodes injected per turn.",
        },
        "value_max_chars": {
            "label": "Value Max Chars",
            "type": "number",
            "default": 300,
            "description": "Per-fact value cap in injected context.",
        },
        "context_max_chars": {
            "label": "Context Max Chars",
            "type": "number",
            "default": 5000,
            "description": "Total injected memory block cap.",
        },
        "write_user_memory": {
            "label": "Write User Memory",
            "type": "checkbox",
            "default": True,
            "description": "Enable writes to durable user docs.",
        },
        "write_room_memory": {
            "label": "Write Room Memory",
            "type": "checkbox",
            "default": True,
            "description": "Enable writes to room docs.",
        },
        "use_people_identities": {
            "label": "Use People Identity Links",
            "type": "checkbox",
            "default": True,
            "description": (
                "Merge linked People into one master memory profile "
                "(Settings > People). Degrades gracefully without the people module."
            ),
        },
        "enable_episodes": {
            "label": "Episodic Memory",
            "type": "checkbox",
            "default": True,
            "description": "Extract and store short episodic memories of events and interactions.",
        },
        "episode_cap": {
            "label": "Episodes Per Person",
            "type": "number",
            "default": 200,
            "description": "Max episodes per person; lowest importance x recency evicted.",
        },
        "fact_half_life_days": {
            "label": "Fact Recency Half-Life (days)",
            "type": "number",
            "default": 120,
            "description": "Recency decay applied to facts during relevance ranking.",
        },
        "episode_half_life_days": {
            "label": "Episode Recency Half-Life (days)",
            "type": "number",
            "default": 14,
            "description": "Recency decay applied to episodes during relevance ranking.",
        },
        "enable_consolidation": {
            "label": "Nightly Consolidation",
            "type": "checkbox",
            "default": True,
            "description": "Nightly LLM-proposed consolidation of duplicate facts plus a profile card.",
        },
        "consolidation_hour": {
            "label": "Consolidation Hour",
            "type": "number",
            "default": 3,
            "description": "Local hour (0-23) at which nightly consolidation runs.",
        },
        "profile_card_max_chars": {
            "label": "Profile Card Max Chars",
            "type": "number",
            "default": 600,
            "description": "Character cap for the generated per-person profile card.",
        },
        "custom_synonyms": {
            "label": "Custom Concept Groups",
            "type": "textarea",
            "default": "",
            "description": (
                "Extra concept groups for retrieval, one group per line, comma-separated words, "
                "e.g. 'coffee, espresso, latte, caffeine'. Extends the built-in concept map."
            ),
        },
        "inject_sensitive_categories": {
            "label": "Inject Sensitive Categories",
            "type": "checkbox",
            "default": False,
            "description": (
                "If off, sensitive facts are stored but injected only when the query strongly "
                "matches them."
            ),
        },
        "enable_inferred_habits": {
            "label": "Inferred Habits (experimental)",
            "type": "checkbox",
            "default": False,
            "description": "Derive low-confidence habit facts from other cores' history (stretch, phase 3).",
        },
    },
    "tags": TAGS,
}

CORE_WEBUI_TAB = {
    "label": "Memory+",
    "order": 21,
    "requires_running": False,
}

# Task lock for read-modify-write on core-owned docs: tool calls are async,
# the extraction loop runs in a thread; both touch the same Redis docs.
_DOC_TASK_LOCK = threading.Lock()


def _settings(redis_obj: Any = None) -> Dict[str, Any]:
    store = redis_obj if redis_obj is not None else redis_client
    return dict(store.hgetall(SETTINGS_KEY) or {}) if store else {}


def _setting_float(settings: Dict[str, Any], key: str, default: float, *, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
    try:
        value = float(settings.get(key, default))
    except Exception:
        value = default
    if value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


def run(stop_event=None) -> None:
    """Background extraction/consolidation loop (cooperative; exits within ~2s of stop)."""
    logger.info("[memory_plus_core] started v%s", __version__)
    try:
        while not (stop_event and stop_event.is_set()):
            interval = _setting_float(_settings(), "interval_seconds", 120.0, min_value=10.0)
            if stop_event:
                if stop_event.wait(interval):
                    break
            else:
                time.sleep(interval)
    except Exception:
        logger.exception("[memory_plus_core] run loop crashed")
    finally:
        logger.info("[memory_plus_core] stopped")


def get_hydra_kernel_tools(*, platform: str = "", **_kwargs) -> List[Dict[str, Any]]:
    """Kernel tools owned by this core (Phase 2: memory_recall, memory_add, memory_remove)."""
    return []


async def run_hydra_kernel_tool(*, tool_id: str, **_kwargs) -> Optional[Dict[str, Any]]:
    """Dispatch a kernel tool id; return None for ids this core does not own."""
    return None


def get_hydra_memory_context_payload(*, platform: str = "", **_kwargs) -> Dict[str, Any]:
    """Hot-path hook: context payload for the Hydra planner. Never raises; LLM-free."""
    try:
        return {}
    except Exception:
        return {}


def get_hydra_system_prompt_fragments(*, role: str = "", **_kwargs) -> Dict[str, List[str]]:
    """Hot-path hook: prompt fragments. Never raises; LLM-free."""
    try:
        return {}
    except Exception:
        return {}


def get_htmlui_tab_data(*, redis_client=None, **_kwargs) -> Dict[str, Any]:
    return {"summary": {}, "items": [], "empty_message": "No memory yet."}


def handle_htmlui_tab_action(*, action: str, payload: Dict[str, Any], redis_client=None, **_kwargs) -> Dict[str, Any]:
    raise KeyError(f"Unsupported action: {action}")


def wipe_memory_plus_core_data() -> int:
    """Delete only core-owned keys (memory_plus_core:*) plus the settings key."""
    deleted = 0
    with _DOC_TASK_LOCK:
        for prefix in (USER_DOC_PREFIX, ROOM_DOC_PREFIX, CURSOR_PREFIX, BACKUP_PREFIX):
            try:
                keys = list(redis_client.scan_iter(f"{prefix}:*"))
            except Exception:
                keys = []
            for key in keys:
                try:
                    redis_client.delete(key)
                    deleted += 1
                except Exception:
                    logger.exception("[memory_plus_core] wipe failed for %s", key)
        try:
            redis_client.delete(META_KEY)
            deleted += 1
        except Exception:
            logger.exception("[memory_plus_core] wipe failed for %s", META_KEY)
    return deleted
