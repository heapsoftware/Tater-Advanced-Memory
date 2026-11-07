"""Memory Plus — enhanced memory Core for Tater.

A fork of the stock Memory Core (``cores/memory_core.py``, TaterTotterson/Tater_Shop)
with pseudo-semantic lexical retrieval, correctable fact updates, first-run
backfill, custom concept synonyms, LLM-proposed nightly consolidation, and
stock-core coexistence (shadow mode / import).

Single-file Core: stdlib + modules Tater already ships. No new pip dependencies.
"""

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import re
import threading
import time
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from helpers import extract_json, get_llm_client_from_env, llm_structured_chat, redis_client
try:
    from helpers import get_primary_llm_client_from_env as _get_primary_llm_client_from_env
except Exception:  # pragma: no cover - compatibility with older Tater runtimes.
    _get_primary_llm_client_from_env = get_llm_client_from_env

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv ships with Tater.
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

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


_MEMORY_TASK_LOCK = threading.Lock()
_MEMORY_TASK_STARTED_AT = 0.0


def _load_settings() -> Dict[str, Any]:
    settings = redis_client.hgetall(SETTINGS_KEY) or {}
    # Legacy keys from earlier drafts are dropped on sight.
    for legacy in ("enabled_platforms", "default_ttl_sec", "allow_new_keys", "allow_new_room_keys", "auto_link_identities", "extraction_max_tokens"):
        if legacy in settings:
            try:
                redis_client.hdel(SETTINGS_KEY, legacy)
            except Exception:
                pass
    return settings


from typing import Any, Dict, List, Optional


MEMORY_PLUS_SCHEMA_VERSION = 2
MEMORY_PLUS_USER_PREFIX = "memory_plus_core:user"
MEMORY_PLUS_ROOM_PREFIX = "memory_plus_core:room"
MEMORY_PLUS_CURSOR_PREFIX = "memory_plus_core:cursor"
MEMORY_PLUS_IDENTITY_DOC_PLATFORM = "identity"
MEMORY_PLUS_IDENTITY_ALIAS_PREFIX = "memory_plus_core:identity_alias"
MEMORY_PLUS_IDENTITY_NAME_PREFIX = "memory_plus_core:identity_name"

_SEGMENT_RE = re.compile(r"[^a-z0-9_.:\-]+")
_FACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTITY_PLACEHOLDER_NAMES = {
    "assistant",
    "bot",
    "user",
    "unknown",
    "unknown_user",
    "telegram_user",
    "discord_user",
    "matrix_user",
    "irc_user",
    "webui_user",
    "macos_user",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0, min_value: int = 0, max_value: Optional[int] = None) -> int:
    try:
        out = int(float(value))
    except Exception:
        out = int(default)
    if out < min_value:
        out = min_value
    if max_value is not None and out > max_value:
        out = max_value
    return out


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return _as_text(value)


def normalize_segment(value: Any, *, default: str = "unknown") -> str:
    raw = _as_text(value).strip().lower()
    if not raw:
        raw = default
    cleaned = _SEGMENT_RE.sub("_", raw).strip("_")
    return cleaned or default


def normalize_fact_key(value: Any) -> str:
    raw = _as_text(value).strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    if not raw:
        return ""
    if raw[0].isdigit():
        raw = f"fact_{raw}"
    if len(raw) > 64:
        raw = raw[:64].rstrip("_")
    if not _FACT_KEY_RE.fullmatch(raw):
        return ""
    return raw


def user_doc_key(platform: Any, user_id: Any) -> str:
    return f"{MEMORY_PLUS_USER_PREFIX}:{normalize_segment(platform, default='webui')}:{normalize_segment(user_id)}"


def room_doc_key(platform: Any, room_id: Any) -> str:
    return f"{MEMORY_PLUS_ROOM_PREFIX}:{normalize_segment(platform, default='webui')}:{normalize_segment(room_id, default='chat')}"


def cursor_key(platform: Any, scope_id: Any) -> str:
    return f"{MEMORY_PLUS_CURSOR_PREFIX}:{normalize_segment(platform, default='webui')}:{normalize_segment(scope_id, default='chat')}"


def identity_doc_key(identity_id: Any) -> str:
    return user_doc_key(MEMORY_PLUS_IDENTITY_DOC_PLATFORM, identity_id)


def identity_alias_key(platform: Any, user_id: Any) -> str:
    return (
        f"{MEMORY_PLUS_IDENTITY_ALIAS_PREFIX}:"
        f"{normalize_segment(platform, default='webui')}:"
        f"{normalize_segment(user_id)}"
    )


def identity_name_key(name: Any) -> str:
    return f"{MEMORY_PLUS_IDENTITY_NAME_PREFIX}:{normalize_segment(name)}"


def normalize_identity_name(value: Any) -> str:
    raw = _as_text(value).strip()
    if raw.startswith("@"):
        raw = raw[1:].strip()
    if ":" in raw and re.fullmatch(r"[A-Za-z0-9._\-]+:[A-Za-z0-9._\-]+", raw):
        local, _, _ = raw.partition(":")
        raw = local.strip() or raw
    normalized = normalize_segment(raw, default="")
    if not normalized:
        return ""
    if normalized in _IDENTITY_PLACEHOLDER_NAMES:
        return ""
    if re.fullmatch(r"-?\d+", normalized):
        return ""
    return normalized


def default_identity_id(
    platform: Any,
    user_id: Any,
    *,
    display_name: Any = None,
) -> str:
    preferred = normalize_identity_name(display_name if display_name is not None else user_id)
    if preferred:
        return preferred
    return (
        f"{normalize_segment(platform, default='webui')}_"
        f"{normalize_segment(user_id)}"
    )


def resolve_identity_id(
    redis_client: Any,
    platform: Any,
    user_id: Any,
    *,
    create: bool = False,
    display_name: Any = None,
    auto_link_name: bool = False,
) -> str:
    platform_seg = normalize_segment(platform, default="webui")
    user_seg = normalize_segment(user_id, default="")
    if not user_seg:
        return ""

    alias_k = identity_alias_key(platform_seg, user_seg)
    alias_raw = ""
    try:
        alias_raw = _as_text(redis_client.get(alias_k)).strip()
    except Exception:
        alias_raw = ""
    if alias_raw:
        alias_id = normalize_segment(alias_raw, default="")
        if alias_id:
            return alias_id

    name_token = normalize_identity_name(display_name if display_name is not None else user_id)
    if auto_link_name and name_token:
        name_k = identity_name_key(name_token)
        try:
            name_raw = _as_text(redis_client.get(name_k)).strip()
        except Exception:
            name_raw = ""
        if name_raw:
            linked_id = normalize_segment(name_raw, default="")
            if linked_id:
                if create:
                    try:
                        redis_client.set(alias_k, linked_id)
                    except Exception:
                        pass
                return linked_id

    if not create:
        return ""

    if auto_link_name and name_token:
        identity_id = default_identity_id(platform_seg, user_seg, display_name=name_token)
    else:
        # Keep identities isolated per-platform unless name auto-linking is enabled.
        identity_id = f"{platform_seg}_{user_seg}"
    try:
        redis_client.set(alias_k, identity_id)
    except Exception:
        pass

    if auto_link_name and name_token:
        name_k = identity_name_key(name_token)
        try:
            existing_name_raw = _as_text(redis_client.get(name_k)).strip()
        except Exception:
            existing_name_raw = ""
        existing_id = normalize_segment(existing_name_raw, default="") if existing_name_raw else ""
        if existing_id:
            identity_id = existing_id
            try:
                redis_client.set(alias_k, identity_id)
            except Exception:
                pass
        else:
            try:
                redis_client.set(name_k, identity_id)
            except Exception:
                pass

    return identity_id


def resolve_user_doc_key(
    redis_client: Any,
    platform: Any,
    user_id: Any,
    *,
    create: bool = False,
    display_name: Any = None,
    auto_link_name: bool = False,
) -> str:
    identity_id = resolve_identity_id(
        redis_client,
        platform,
        user_id,
        create=create,
        display_name=display_name,
        auto_link_name=auto_link_name,
    )
    if identity_id:
        return identity_doc_key(identity_id)
    user_seg = normalize_segment(user_id, default="")
    if not user_seg:
        return ""
    return user_doc_key(platform, user_seg)


def set_identity_alias(
    redis_client: Any,
    platform: Any,
    user_id: Any,
    identity_id: Any,
    *,
    display_name: Any = None,
    auto_link_name: bool = True,
) -> str:
    platform_seg = normalize_segment(platform, default="webui")
    user_seg = normalize_segment(user_id, default="")
    identity_seg = normalize_segment(identity_id, default="")
    if not platform_seg or not user_seg or not identity_seg:
        return ""
    alias_k = identity_alias_key(platform_seg, user_seg)
    try:
        redis_client.set(alias_k, identity_seg)
    except Exception:
        return ""
    if auto_link_name:
        name_token = normalize_identity_name(display_name if display_name is not None else user_id)
        if name_token:
            try:
                redis_client.set(identity_name_key(name_token), identity_seg)
            except Exception:
                pass
    return identity_seg


def clear_identity_alias(redis_client: Any, platform: Any, user_id: Any) -> int:
    key = identity_alias_key(platform, user_id)
    try:
        return int(redis_client.delete(key) or 0)
    except Exception:
        return 0


def list_identity_aliases(redis_client: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        for raw_key in redis_client.scan_iter(match=f"{MEMORY_PLUS_IDENTITY_ALIAS_PREFIX}:*", count=500):
            key = _as_text(raw_key).strip()
            if not key:
                continue
            payload = key.split(f"{MEMORY_PLUS_IDENTITY_ALIAS_PREFIX}:", 1)[-1]
            platform_seg, sep, user_seg = payload.partition(":")
            if not sep or not platform_seg or not user_seg:
                continue
            try:
                raw_identity = _as_text(redis_client.get(key)).strip()
            except Exception:
                raw_identity = ""
            identity_seg = normalize_segment(raw_identity, default="")
            if not identity_seg:
                continue
            rows.append(
                {
                    "platform": platform_seg,
                    "user_id": user_seg,
                    "identity_id": identity_seg,
                    "alias_key": key,
                    "doc_key": identity_doc_key(identity_seg),
                }
            )
    except Exception:
        return []

    rows.sort(key=lambda row: (row.get("identity_id") or "", row.get("platform") or "", row.get("user_id") or ""))
    return rows


def _value_fingerprint(value: Any) -> str:
    try:
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return _as_text(value)


def _coerce_evidence(raw: Any) -> List[str]:
    if isinstance(raw, list):
        source = raw
    elif isinstance(raw, tuple):
        source = list(raw)
    else:
        source = []

    out: List[str] = []
    for item in source:
        text = _as_text(item).strip()
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out[:12]


def _coerce_confidence(raw: Any) -> float:
    value = _safe_float(raw, default=0.0)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _coerce_ttl(raw: Any, default_ttl_sec: int) -> int:
    default_i = _safe_int(default_ttl_sec, default=0, min_value=0, max_value=31_536_000)
    return _safe_int(raw, default=default_i, min_value=0, max_value=31_536_000)


def _coerce_retired_facts(raw: Any) -> List[Dict[str, Any]]:
    source = raw if isinstance(raw, list) else []
    out: List[Dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        fact_key = normalize_fact_key(item.get("key") or item.get("candidate_key"))
        if not fact_key or "value" not in item:
            continue
        valid_from = _safe_float(item.get("valid_from"), default=_safe_float(item.get("updated_at"), 0.0))
        valid_to = _safe_float(item.get("valid_to"), default=_safe_float(item.get("superseded_at"), 0.0))
        out.append(
            {
                "key": fact_key,
                "value": _json_safe(item.get("value")),
                "confidence": _coerce_confidence(item.get("confidence")),
                "evidence": _coerce_evidence(item.get("evidence")),
                "valid_from": valid_from,
                "valid_to": valid_to,
                "reason": _as_text(item.get("reason") or "superseded").strip() or "superseded",
            }
        )
    return out[-100:]


def empty_memory_doc(*, now: Optional[float] = None) -> Dict[str, Any]:
    ts = float(now if now is not None else time.time())
    return {
        "schema_version": MEMORY_PLUS_SCHEMA_VERSION,
        "last_updated": ts,
        "facts": {},
        "retired_facts": [],
    }


def parse_memory_doc(raw: Any, *, now: Optional[float] = None) -> Dict[str, Any]:
    ts_now = float(now if now is not None else time.time())
    if isinstance(raw, dict):
        parsed = dict(raw)
    else:
        text = _as_text(raw).strip()
        if not text:
            return empty_memory_doc(now=ts_now)

        try:
            parsed = json.loads(text)
        except Exception:
            return empty_memory_doc(now=ts_now)

    if not isinstance(parsed, dict):
        return empty_memory_doc(now=ts_now)

    facts_in = parsed.get("facts")
    facts_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(facts_in, dict):
        for raw_key, raw_fact in facts_in.items():
            fact_key = normalize_fact_key(raw_key)
            if not fact_key or not isinstance(raw_fact, dict):
                continue
            value = _json_safe(raw_fact.get("value"))
            confidence = _coerce_confidence(raw_fact.get("confidence"))
            # Platform memory facts are durable; TTL is intentionally disabled.
            ttl_sec = 0
            evidence = _coerce_evidence(raw_fact.get("evidence"))
            updated_at = _safe_float(raw_fact.get("updated_at"), default=0.0)
            valid_from = _safe_float(raw_fact.get("valid_from"), default=updated_at)
            last_confirmed_at = _safe_float(raw_fact.get("last_confirmed_at"), default=updated_at)
            facts_map[fact_key] = {
                "value": value,
                "confidence": confidence,
                "evidence": evidence,
                "ttl_sec": ttl_sec,
                "updated_at": updated_at,
                "valid_from": valid_from,
                "last_confirmed_at": last_confirmed_at,
                "expires_at": None,
            }

    return {
        "schema_version": MEMORY_PLUS_SCHEMA_VERSION,
        "last_updated": _safe_float(parsed.get("last_updated"), default=ts_now),
        "facts": facts_map,
        "retired_facts": _coerce_retired_facts(parsed.get("retired_facts")),
    }


def fact_is_expired(fact: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    # Expiration is disabled for platform memory.
    return False


def prune_expired_facts(doc: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    ts_now = float(now if now is not None else time.time())
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        doc["facts"] = {}
        return False

    expired = [key for key, fact in facts.items() if isinstance(fact, dict) and fact_is_expired(fact, now=ts_now)]
    if not expired:
        return False

    for key in expired:
        facts.pop(key, None)
    doc["last_updated"] = ts_now
    return True


def memory_doc_to_json(doc: Dict[str, Any], *, now: Optional[float] = None) -> str:
    ts_now = float(now if now is not None else time.time())
    parsed = parse_memory_doc(doc, now=ts_now)
    prune_expired_facts(parsed, now=ts_now)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def merge_observation(
    doc: Dict[str, Any],
    observation: Dict[str, Any],
    *,
    min_confidence: float,
    default_ttl_sec: int,
    allow_new_keys: bool,
    now: Optional[float] = None,
) -> bool:
    """Merge one extracted observation into a doc under correction-safe rules.

    Fixes the stock-confidence-ratchet bug: differing values are not blocked by
    an already-matured confidence. See MEMORY_PLUS_CORE_SPEC.md section 7.4.
    """
    ts_now = float(now if now is not None else time.time())
    if not isinstance(observation, dict):
        return False

    fact_key = normalize_fact_key(observation.get("candidate_key") or observation.get("key"))
    if not fact_key:
        return False

    if "value" not in observation:
        return False
    value = _json_safe(observation.get("value"))
    confidence = _coerce_confidence(observation.get("confidence"))
    if confidence < float(min_confidence):
        return False

    source = _as_text(observation.get("source")).strip().lower() or (
        "stated" if bool(observation.get("stated")) else "inferred"
    )
    if source not in {"stated", "inferred", "imported", "manual"}:
        source = "inferred"
    evidence = _coerce_evidence(
        observation.get("evidence") or observation.get("evidence_message_ids")
    )

    facts = doc.get("facts")
    if not isinstance(facts, dict):
        facts = {}
        doc["facts"] = facts

    if not allow_new_keys and fact_key not in facts:
        return False

    def _new_fact(value_in: Any) -> Dict[str, Any]:
        return {
            "value": _json_safe(value_in),
            "confidence": confidence,
            "evidence": evidence,
            "updated_at": ts_now,
            "first_seen": ts_now,
            "mentions": 1,
            "source": source,
            "category": _as_text(observation.get("category")).strip().lower(),
            "previous": [],
            "pending_alt": None,
        }

    existing = facts.get(fact_key)
    if not isinstance(existing, dict):
        facts[fact_key] = _new_fact(value)
        doc["last_updated"] = ts_now
        return True

    existing_conf = _coerce_confidence(existing.get("confidence"))
    same_value = _value_fingerprint(existing.get("value")) == _value_fingerprint(value)

    # ---- Same value: bump mentions, merge evidence, modest confidence bump.
    if same_value:
        existing_evidence = _coerce_evidence(existing.get("evidence"))
        added = [item for item in evidence if item not in existing_evidence]
        existing["evidence"] = existing_evidence + added
        existing["mentions"] = _safe_int(existing.get("mentions"), 1, min_value=1) + 1
        merged_conf = max(existing_conf, confidence)
        if added:
            merged_conf = min(0.99, max(merged_conf, existing_conf + min(0.06, 0.02 * len(added))))
        # A repeated confirmation validates an earlier alt value; clear any pending alternate.
        existing["pending_alt"] = None
        existing["confidence"] = merged_conf
        existing["updated_at"] = ts_now
        if not _as_text(existing.get("source")).strip() and source == "stated":
            existing["source"] = "stated"
        doc["last_updated"] = ts_now
        return True

    # ---- List-valued facts: merge as a set union unless the observation supersedes.
    is_list = isinstance(existing.get("value"), list) and isinstance(value, list)
    if is_list and not bool(observation.get("supersedes")):
        union = []
        seen: set = set()
        for item in list(existing.get("value") or []) + list(value):
            token = _value_fingerprint(item)
            if token in seen:
                continue
            seen.add(token)
            union.append(item)
        existing["value"] = _json_safe(union)
        existing["confidence"] = min(0.99, max(existing_conf, confidence))
        existing["mentions"] = _safe_int(existing.get("mentions"), 1, min_value=1) + 1
        ev = _coerce_evidence(existing.get("evidence") + [x for x in evidence if x not in _coerce_evidence(existing.get("evidence"))])
        existing["evidence"] = ev
        existing["updated_at"] = ts_now
        doc["last_updated"] = ts_now
        return True

    # ---- Differing value: decide replace now vs. pend for a second sighting.
    explicit = bool(observation.get("supersedes")) or _as_text(observation.get("operation")).strip().lower() in {
        "replace", "correct", "supersede"
    }
    age_days = (ts_now - _safe_float(existing.get("updated_at"), default=ts_now)) / 86_400.0
    old_fact_age_days = (ts_now - _safe_float(existing.get("first_seen"), default=_safe_float(existing.get("updated_at"), default=ts_now))) / 86_400.0

    reason = ""
    if explicit:
        reason = "supersedes"
    elif age_days >= 30 or old_fact_age_days >= 30:
        reason = "age"
    elif confidence >= existing_conf - 0.15:
        reason = "fresh"

    pending = existing.get("pending_alt")
    if reason == "" and isinstance(pending, dict):
        pending_value_same = _value_fingerprint(pending.get("value")) == _value_fingerprint(value)
        count = _safe_int(pending.get("count"), 1, min_value=1)
        if pending_value_same:
            count += 1
            if count >= 2:
                reason = "repeat"
        else:
            pending = {"value": _json_safe(value), "count": 1, "last_seen": ts_now}
            existing["pending_alt"] = pending
            return False
        if reason == "":
            existing["pending_alt"] = {"value": _json_safe(value), "count": count, "last_seen": ts_now}
            return True
    elif reason == "":
        existing["pending_alt"] = {"value": _json_safe(value), "count": 1, "last_seen": ts_now}
        return False

    # ---- Replace: record the previous value, carry useful counters over.
    previous = existing.get("previous")
    if not isinstance(previous, list):
        previous = []
    previous.append(
        {
            "value": _json_safe(existing.get("value")),
            "until": ts_now,
        }
    )
    previous = previous[-3:]

    facts[fact_key] = {
        "value": _json_safe(value),
        "confidence": confidence,
        "evidence": _coerce_evidence(existing.get("evidence")) + [e for e in evidence if e not in _coerce_evidence(existing.get("evidence"))],
        "updated_at": ts_now,
        "first_seen": _safe_float(existing.get("first_seen"), default=ts_now),
        "mentions": _safe_int(existing.get("mentions"), 1, min_value=1) + 1,
        "source": source if source != "inferred" else _as_text(existing.get("source")).strip().lower() or "inferred",
        "category": _as_text(observation.get("category")).strip().lower() or _as_text(existing.get("category")).strip().lower(),
        "previous": previous,
        "pending_alt": None,
    }
    doc["last_updated"] = ts_now
    return True


def merge_doc_facts(
    target_doc: Dict[str, Any],
    source_doc: Dict[str, Any],
    *,
    min_confidence: float = 0.0,
    now: Optional[float] = None,
) -> int:
    ts_now = float(now if now is not None else time.time())
    source = parse_memory_doc(source_doc, now=ts_now)
    source_facts = source.get("facts") if isinstance(source.get("facts"), dict) else {}
    if not source_facts:
        return 0

    changed = 0
    for fact_key, fact in source_facts.items():
        if not isinstance(fact, dict):
            continue
        if "value" not in fact:
            continue
        observation = {
            "candidate_key": fact_key,
            "value": fact.get("value"),
            "confidence": _coerce_confidence(fact.get("confidence")),
            "evidence": _coerce_evidence(fact.get("evidence")),
            "ttl_sec": 0,
        }
        if merge_observation(
            target_doc,
            observation,
            min_confidence=float(min_confidence),
            default_ttl_sec=0,
            allow_new_keys=True,
            now=ts_now,
        ):
            changed += 1
    return changed


def load_doc(redis_client: Any, key: str, *, now: Optional[float] = None) -> Dict[str, Any]:
    ts_now = float(now if now is not None else time.time())
    raw = None
    try:
        raw = redis_client.get(key)
    except Exception:
        raw = None
    doc = parse_memory_doc(raw, now=ts_now)
    if prune_expired_facts(doc, now=ts_now):
        save_doc(redis_client, key, doc, now=ts_now)
    return doc


def save_doc(redis_client: Any, key: str, doc: Dict[str, Any], *, now: Optional[float] = None) -> None:
    ts_now = float(now if now is not None else time.time())
    text = memory_doc_to_json(doc, now=ts_now)
    redis_client.set(key, text)


def forget_fact_keys(doc: Dict[str, Any], keys: List[str]) -> int:
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        facts = {}
        doc["facts"] = facts

    deleted = 0
    for raw_key in keys:
        fact_key = normalize_fact_key(raw_key)
        if not fact_key:
            continue
        if fact_key in facts:
            facts.pop(fact_key, None)
            deleted += 1
        retired = doc.get("retired_facts")
        if isinstance(retired, list):
            before_retired = len(retired)
            doc["retired_facts"] = [
                item for item in retired
                if not isinstance(item, dict) or normalize_fact_key(item.get("key")) != fact_key
            ]
            deleted += before_retired - len(doc["retired_facts"])
    if deleted > 0:
        doc["last_updated"] = time.time()
    return deleted


def retire_fact_keys(
    doc: Dict[str, Any],
    keys: List[str],
    *,
    reason: str = "ended",
    now: Optional[float] = None,
) -> int:
    ts_now = float(now if now is not None else time.time())
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        return 0
    retired = doc.get("retired_facts")
    if not isinstance(retired, list):
        retired = []
    count = 0
    for raw_key in keys:
        fact_key = normalize_fact_key(raw_key)
        fact = facts.get(fact_key) if fact_key else None
        if not fact_key or not isinstance(fact, dict):
            continue
        retired.append(
            {
                "key": fact_key,
                "value": _json_safe(fact.get("value")),
                "confidence": _coerce_confidence(fact.get("confidence")),
                "evidence": _coerce_evidence(fact.get("evidence")),
                "valid_from": _safe_float(fact.get("valid_from"), default=_safe_float(fact.get("updated_at"), ts_now)),
                "valid_to": ts_now,
                "reason": _as_text(reason).strip() or "ended",
            }
        )
        facts.pop(fact_key, None)
        count += 1
    if count:
        doc["retired_facts"] = _coerce_retired_facts(retired)
        doc["last_updated"] = ts_now
    return count


def summarize_doc(
    doc: Dict[str, Any],
    *,
    max_items: int = 6,
    min_confidence: float = 0.0,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    ts_now = float(now if now is not None else time.time())
    parsed = parse_memory_doc(doc, now=ts_now)
    prune_expired_facts(parsed, now=ts_now)
    facts = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {}

    rows: List[Dict[str, Any]] = []
    for key, fact in facts.items():
        if not isinstance(fact, dict):
            continue
        confidence = _coerce_confidence(fact.get("confidence"))
        if confidence < float(min_confidence):
            continue
        rows.append(
            {
                "key": key,
                "value": fact.get("value"),
                "confidence": confidence,
                "evidence": _coerce_evidence(fact.get("evidence")),
                "ttl_sec": _coerce_ttl(fact.get("ttl_sec"), 0),
                "updated_at": _safe_float(fact.get("updated_at"), default=0.0),
                "valid_from": _safe_float(fact.get("valid_from"), default=_safe_float(fact.get("updated_at"), 0.0)),
                "last_confirmed_at": _safe_float(fact.get("last_confirmed_at"), default=_safe_float(fact.get("updated_at"), 0.0)),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row.get("confidence") or 0.0),
            -float(row.get("updated_at") or 0.0),
            str(row.get("key") or ""),
        )
    )
    return rows[: max(1, int(max_items))]


def value_to_text(value: Any, *, max_chars: int = 100) -> str:
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = _as_text(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(8, max_chars - 3)].rstrip() + "..."

_SUPPORTED_PLATFORMS = (
    "webui",
    "macos",
    "discord",
    "telegram",
    "irc",
    "matrix",
    "meshtastic",
    "homeassistant",
    "homekit",
    "xbmc",
    "voice_core",
)
_AUTO_STATE_KEYS = {
    "macos": "macos_portal_running",
    "discord": "discord_portal_running",
    "telegram": "telegram_portal_running",
    "irc": "irc_portal_running",
    "matrix": "matrix_portal_running",
    "meshtastic": "meshtastic_portal_running",
    "homekit": "homekit_portal_running",
    "xbmc": "xbmc_portal_running",
}
_HISTORY_SCAN_SPECS = {
    "macos": ("tater:macos:session:*:history", "tater:macos:session:", ":history"),
    "discord": ("tater:channel:*:history", "tater:channel:", ":history"),
    "telegram": ("tater:telegram:*:history", "tater:telegram:", ":history"),
    "irc": ("tater:irc:*:history", "tater:irc:", ":history"),
    "matrix": ("tater:matrix:*:history", "tater:matrix:", ":history"),
    "meshtastic": ("tater:meshtastic:*:history", "tater:meshtastic:", ":history"),
    "homeassistant": ("tater:ha:session:*:history", "tater:ha:session:", ":history"),
    "homekit": ("tater:homekit:session:*:history", "tater:homekit:session:", ":history"),
    "xbmc": ("tater:xbmc:session:*:history", "tater:xbmc:session:", ":history"),
    "voice_core": ("tater:voice:conv:*:history", "tater:voice:conv:", ":history"),
}
_ROOM_LABEL_PREFIX = "tater:room_label"
_USER_LABEL_PREFIX = "tater:user_label"
_MARKER_SKIP = {"plugin_wait"}
_TRANSIENT_FACT_PREFIXES = (
    "request_",
    "requests_",
    "requested_",
    "asking_",
    "asks_",
    "ask_",
    "follow_up_",
    "followup_",
    "needs_",
    "need_",
    "wants_",
    "want_",
    "do_",
    "did_",
    "is_",
    "are_",
    "was_",
    "were_",
)
_TRANSIENT_FACT_EXACT = {
    "latest_request",
    "current_request",
    "recent_request",
}
_USER_PROFILE_SCHEMA: Dict[str, Tuple[str, ...]] = {
    "demographic_information": (
        "age",
        "gender",
        "ethnicity",
        "occupation",
        "education",
    ),
    "physical_traits": (
        "height",
        "weight",
        "hair_color",
        "eye_color",
        "distinguishing_features",
    ),
    "personality_traits": (
        "temperament",
        "values_and_beliefs",
        "interests",
        "strengths_and_weaknesses",
    ),
    "behavioral_patterns": (
        "communication_style",
        "conflict_resolution",
        "social_interactions",
    ),
    "lifestyle_choices": (
        "daily_routine",
        "health_habits",
        "recreational_activities",
    ),
    "social_media_presence": (
        "platforms_used",
        "content_shared",
        "network",
    ),
    "life_experiences": (
        "major_life_events",
        "achievements",
        "challenges",
    ),
    "family_background": (
        "family_dynamics",
        "cultural_influences",
        "traditions",
    ),
    "mental_and_emotional_health": (
        "mental_well_being",
        "coping_mechanisms",
        "resilience",
    ),
    "goals_and_aspirations": (
        "short_term_goals",
        "long_term_aspirations",
    ),
    "assistant_memory_notes": (
        "user_wants_you_to_remember",
    ),
}


def _room_label_key(platform: Any, room_id: Any) -> str:
    platform_name = normalize_segment(platform, default="unknown")
    scope_id = normalize_segment(room_id, default="")
    if not scope_id:
        return ""
    return f"{_ROOM_LABEL_PREFIX}:{platform_name}:{scope_id}"


def _user_label_key(platform: Any, user_id: Any) -> str:
    platform_name = normalize_segment(platform, default="unknown")
    scope_id = normalize_segment(user_id, default="")
    if not scope_id:
        return ""
    return f"{_USER_LABEL_PREFIX}:{platform_name}:{scope_id}"


def _save_room_label(platform: Any, room_id: Any, label: Any) -> None:
    key = _room_label_key(platform, room_id)
    text = _as_text(label).strip()
    if not key or not text:
        return
    try:
        redis_client.set(key, text)
    except Exception:
        return


def _save_user_label(platform: Any, user_id: Any, label: Any) -> None:
    key = _user_label_key(platform, user_id)
    text = _as_text(label).strip()
    if not key or not text:
        return
    try:
        redis_client.set(key, text)
    except Exception:
        return


def _extract_room_label_from_entry(entry: Dict[str, Any], *, platform: str, scope_id: str) -> str:
    if not isinstance(entry, dict):
        return ""

    candidate_keys = (
        "room_name",
        "room",
        "room_label",
        "channel_name",
        "channel",
        "chat_title",
        "chat_name",
        "title",
        "name",
    )
    for key in candidate_keys:
        val = _as_text(entry.get(key)).strip()
        if val:
            return val

    if platform == "irc":
        scope = _as_text(scope_id).strip()
        if scope.startswith("#"):
            return scope
    return ""


def _remember_scope_labels(platform: str, scope_id: str, messages: List[Dict[str, Any]]) -> None:
    if not isinstance(messages, list):
        return

    for item in messages:
        if not isinstance(item, dict):
            continue
        user_id = _as_text(item.get("user_id")).strip()
        user_name = _as_text(item.get("user_name")).strip()
        role = _as_text(item.get("role")).strip().lower()
        if role == "user" and user_id and user_id.lower() not in {"assistant", "unknown_user"} and user_name:
            _save_user_label(platform, user_id, user_name)

    for item in messages:
        if not isinstance(item, dict):
            continue
        room_name = _as_text(item.get("room_name")).strip()
        if room_name:
            _save_room_label(platform, scope_id, room_name)
            return
_USER_ALLOWED_FACT_KEYS: Tuple[str, ...] = tuple(
    fact_key
    for fields in _USER_PROFILE_SCHEMA.values()
    for fact_key in fields
)
_USER_ALLOWED_FACT_KEY_SET = set(_USER_ALLOWED_FACT_KEYS)
_USER_SCHEMA_PROMPT_TEXT = "\n".join(
    f"- {category.replace('_', ' ').title()}: {', '.join(fields)}"
    for category, fields in _USER_PROFILE_SCHEMA.items()
)
_USER_ALLOWED_FACT_KEYS_PROMPT = ", ".join(_USER_ALLOWED_FACT_KEYS)
_ROOM_PROFILE_SCHEMA: Dict[str, Tuple[str, ...]] = {
    "room_identity": (
        "room_purpose",
        "shared_topics",
        "shared_projects",
        "inside_jokes",
    ),
    "communication_norms": (
        "communication_tone",
        "response_style",
        "conflict_norms",
    ),
    "shared_defaults": (
        "default_units",
        "default_timezone",
        "default_language",
        "decision_preferences",
    ),
    "collaboration_context": (
        "active_goals",
        "recurring_tasks",
        "shared_tools",
        "reference_links",
    ),
    "constraints_and_policies": (
        "shared_constraints",
        "moderation_preferences",
        "privacy_expectations",
    ),
}
_ROOM_ALLOWED_FACT_KEYS: Tuple[str, ...] = tuple(
    fact_key
    for fields in _ROOM_PROFILE_SCHEMA.values()
    for fact_key in fields
)
_ROOM_ALLOWED_FACT_KEY_SET = set(_ROOM_ALLOWED_FACT_KEYS)
_ROOM_SCHEMA_PROMPT_TEXT = "\n".join(
    f"- {category.replace('_', ' ').title()}: {', '.join(fields)}"
    for category, fields in _ROOM_PROFILE_SCHEMA.items()
)
_ROOM_ALLOWED_FACT_KEYS_PROMPT = ", ".join(_ROOM_ALLOWED_FACT_KEYS)
_DYNAMIC_FACT_KEY_BLOCKLIST_EXACT = {
    "api_key",
    "auth_token",
    "credential",
    "credentials",
    "credit_card",
    "current_task",
    "current_user_request",
    "cvv",
    "jwt",
    "latest_message",
    "latest_user_request",
    "misc",
    "note",
    "notes",
    "otp",
    "password",
    "pin",
    "private_key",
    "recent_message",
    "recent_user_request",
    "secret",
    "session_token",
    "social_security_number",
    "ssn",
    "token",
}
_DYNAMIC_FACT_KEY_BLOCKLIST_PARTS = (
    "api_key",
    "auth_token",
    "credential",
    "credit_card",
    "current_request",
    "jwt",
    "latest_request",
    "one_time_code",
    "password",
    "private_key",
    "recent_request",
    "session_token",
    "social_security",
)
_FACT_KEY_ALIASES = {
    "favorite_colour": "favorite_color",
    "favourite_color": "favorite_color",
    "favourite_colour": "favorite_color",
    "job": "occupation",
    "profession": "occupation",
    "communication_preferences": "communication_style",
    "short_term_goal": "short_term_goals",
    "long_term_goal": "long_term_aspirations",
    "long_term_goals": "long_term_aspirations",
    "remembered_notes": "user_wants_you_to_remember",
    "preferred_language": "default_language",
    "timezone": "default_timezone",
}


def _is_transient_fact_key(fact_key: Any) -> bool:
    key = normalize_fact_key(fact_key)
    if not key:
        return False
    if key in _TRANSIENT_FACT_EXACT:
        return True
    return any(key.startswith(prefix) for prefix in _TRANSIENT_FACT_PREFIXES)


def _is_safe_dynamic_fact_key(fact_key: Any) -> bool:
    key = normalize_fact_key(fact_key)
    if not key:
        return False
    if key in _USER_ALLOWED_FACT_KEY_SET or key in _ROOM_ALLOWED_FACT_KEY_SET:
        return True
    if len(key) < 3:
        return False
    if _is_transient_fact_key(key):
        return False
    if key in _DYNAMIC_FACT_KEY_BLOCKLIST_EXACT:
        return False
    if any(part in key for part in _DYNAMIC_FACT_KEY_BLOCKLIST_PARTS):
        return False
    return True


def _canonical_user_fact_key(raw_key: Any) -> str:
    fact_key = normalize_fact_key(raw_key)
    fact_key = _FACT_KEY_ALIASES.get(fact_key, fact_key)
    if not fact_key:
        return ""
    if fact_key in _USER_ALLOWED_FACT_KEY_SET:
        return fact_key
    if _is_safe_dynamic_fact_key(fact_key):
        return fact_key
    return ""


def _canonical_room_fact_key(raw_key: Any) -> str:
    fact_key = normalize_fact_key(raw_key)
    fact_key = _FACT_KEY_ALIASES.get(fact_key, fact_key)
    if not fact_key:
        return ""
    if fact_key in _ROOM_ALLOWED_FACT_KEY_SET:
        return fact_key
    if _is_safe_dynamic_fact_key(fact_key):
        return fact_key
    return ""


def _scrub_transient_facts(doc: Dict[str, Any], *, now: Optional[float] = None) -> int:
    if not isinstance(doc, dict):
        return 0
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        return 0
    drop = [key for key in list(facts.keys()) if _is_transient_fact_key(key)]
    if not drop:
        return 0
    for key in drop:
        facts.pop(key, None)
    doc["last_updated"] = float(now if now is not None else time.time())
    return len(drop)


def _scrub_non_schema_user_facts(doc: Dict[str, Any], *, now: Optional[float] = None) -> int:
    if not isinstance(doc, dict):
        return 0
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        return 0
    drop = [key for key in list(facts.keys()) if not _is_safe_dynamic_fact_key(key)]
    if not drop:
        return 0
    for key in drop:
        facts.pop(key, None)
    doc["last_updated"] = float(now if now is not None else time.time())
    return len(drop)


def _scrub_non_schema_room_facts(doc: Dict[str, Any], *, now: Optional[float] = None) -> int:
    if not isinstance(doc, dict):
        return 0
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        return 0
    drop = [key for key in list(facts.keys()) if not _is_safe_dynamic_fact_key(key)]
    if not drop:
        return 0
    for key in drop:
        facts.pop(key, None)
    doc["last_updated"] = float(now if now is not None else time.time())
    return len(drop)


_MEMORY_AUX_SCHEMA_VERSION = 1
_EPISODE_KINDS = {"decision", "outcome", "milestone", "project", "commitment", "experience"}
_EPISODE_STATUSES = {"open", "resolved", "completed", "cancelled", "historical"}
_PROFILE_CARD_PRIORITY = (
    "communication_style",
    "response_style",
    "occupation",
    "interests",
    "values_and_beliefs",
    "short_term_goals",
    "long_term_aspirations",
    "daily_routine",
    "recreational_activities",
    "user_wants_you_to_remember",
)
_SENSITIVE_FACT_PARTS = (
    "ethnicity",
    "mental",
    "health",
    "medical",
    "diagnosis",
    "religion",
    "politic",
    "sexual",
    "ssn",
    "credit_card",
)


def _memory_aux_key(kind: str, owner_doc_key: str) -> str:
    owner = _as_text(owner_doc_key).strip()
    if owner.startswith("memory_plus_core:"):
        owner = owner[4:]
    owner = owner or "unknown"
    return f"memory_plus_core:aux:{normalize_segment(kind, default='aux')}:{owner}"


def _memory_load_json(redis_obj: Any, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw = redis_obj.get(key)
    except Exception:
        raw = None
    if isinstance(raw, dict):
        parsed = dict(raw)
    else:
        try:
            parsed = json.loads(_as_text(raw)) if _as_text(raw).strip() else {}
        except Exception:
            parsed = {}
    return parsed if isinstance(parsed, dict) else dict(default)


def _memory_save_json(redis_obj: Any, key: str, payload: Dict[str, Any]) -> None:
    redis_obj.set(key, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _memory_clean_text(value: Any, max_chars: int = 500) -> str:
    text = " ".join(_as_text(value).split()).strip()
    if len(text) > max_chars:
        return text[: max(8, max_chars - 3)].rstrip() + "..."
    return text


def _memory_string_list(value: Any, *, max_items: int = 12, max_chars: int = 160) -> List[str]:
    source = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
    out: List[str] = []
    seen: set[str] = set()
    for item in source:
        text = _memory_clean_text(item, max_chars=max_chars)
        token = text.lower()
        if not text or token in seen:
            continue
        seen.add(token)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _memory_episode_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9']+", _as_text(value).lower()))


def _memory_episode_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_tokens = _memory_episode_tokens(left.get("summary"))
    right_tokens = _memory_episode_tokens(right.get("summary"))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _memory_normalize_episode(raw: Dict[str, Any], *, now_ts: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    ts_now = float(now_ts if now_ts is not None else time.time())
    summary = _memory_clean_text(raw.get("summary"), max_chars=700)
    if not summary:
        return None
    kind = _as_text(raw.get("kind")).strip().lower()
    if kind not in _EPISODE_KINDS:
        kind = "experience"
    status = _as_text(raw.get("status")).strip().lower()
    if status not in _EPISODE_STATUSES:
        status = "open" if kind in {"project", "commitment"} else "historical"
    occurred_at = _as_float(raw.get("occurred_at"), ts_now, min_value=0.0)
    evidence = _memory_string_list(raw.get("evidence") or raw.get("source_message_ids"), max_items=12, max_chars=160)
    fingerprint_source = json.dumps(
        {"summary": summary.lower(), "kind": kind, "evidence": evidence[:3]},
        ensure_ascii=False,
        sort_keys=True,
    )
    episode_id = _as_text(raw.get("id")).strip() or hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:20]
    return {
        "id": episode_id,
        "kind": kind,
        "summary": summary,
        "topics": _memory_string_list(raw.get("topics"), max_items=10, max_chars=80),
        "outcome": _memory_clean_text(raw.get("outcome"), max_chars=400),
        "commitments": _memory_string_list(raw.get("commitments"), max_items=10, max_chars=240),
        "status": status,
        "salience": _as_float(raw.get("salience"), 0.0, min_value=0.0, max_value=1.0),
        "occurred_at": occurred_at,
        "created_at": _as_float(raw.get("created_at"), ts_now, min_value=0.0),
        "updated_at": _as_float(raw.get("updated_at"), ts_now, min_value=0.0),
        "evidence": evidence,
    }


def _memory_load_episodes(redis_obj: Any, owner_doc_key: str) -> List[Dict[str, Any]]:
    payload = _memory_load_json(
        redis_obj,
        _memory_aux_key("episodes", owner_doc_key),
        {"schema_version": _MEMORY_AUX_SCHEMA_VERSION, "episodes": []},
    )
    source = payload.get("episodes") if isinstance(payload.get("episodes"), list) else []
    out: List[Dict[str, Any]] = []
    for item in source:
        normalized = _memory_normalize_episode(item) if isinstance(item, dict) else None
        if normalized:
            out.append(normalized)
    return out


def _memory_store_episodes(
    redis_obj: Any,
    owner_doc_key: str,
    episodes: List[Dict[str, Any]],
    *,
    min_salience: float,
    max_items: int,
    now_ts: Optional[float] = None,
) -> int:
    ts_now = float(now_ts if now_ts is not None else time.time())
    stored = _memory_load_episodes(redis_obj, owner_doc_key)
    changed = 0
    for raw in episodes:
        episode = _memory_normalize_episode(raw, now_ts=ts_now)
        if not episode or float(episode.get("salience") or 0.0) < float(min_salience):
            continue
        match: Optional[Dict[str, Any]] = None
        for candidate in stored:
            if candidate.get("id") == episode.get("id") or _memory_episode_similarity(candidate, episode) >= 0.84:
                match = candidate
                break
        if match is None:
            stored.append(episode)
            changed += 1
            continue
        before = json.dumps(match, ensure_ascii=False, sort_keys=True)
        match["topics"] = _memory_string_list(list(match.get("topics") or []) + list(episode.get("topics") or []), max_items=10, max_chars=80)
        match["commitments"] = _memory_string_list(list(match.get("commitments") or []) + list(episode.get("commitments") or []), max_items=10, max_chars=240)
        match["evidence"] = _memory_string_list(list(match.get("evidence") or []) + list(episode.get("evidence") or []), max_items=12, max_chars=160)
        match["salience"] = max(float(match.get("salience") or 0.0), float(episode.get("salience") or 0.0))
        if episode.get("outcome"):
            match["outcome"] = episode.get("outcome")
        if episode.get("status") != "historical" or match.get("status") == "historical":
            match["status"] = episode.get("status")
        match["updated_at"] = ts_now
        if json.dumps(match, ensure_ascii=False, sort_keys=True) != before:
            changed += 1
    if not changed:
        return 0
    stored.sort(
        key=lambda item: (
            float(item.get("salience") or 0.0),
            float(item.get("occurred_at") or 0.0),
        ),
        reverse=True,
    )
    bounded = stored[: max(25, int(max_items))]
    _memory_save_json(
        redis_obj,
        _memory_aux_key("episodes", owner_doc_key),
        {"schema_version": _MEMORY_AUX_SCHEMA_VERSION, "owner_doc_key": owner_doc_key, "updated_at": ts_now, "episodes": bounded},
    )
    return changed


def _memory_delete_episodes(redis_obj: Any, owner_doc_key: str, episode_ids: List[str]) -> int:
    targets = {_as_text(item).strip() for item in episode_ids if _as_text(item).strip()}
    if not targets:
        return 0
    episodes = _memory_load_episodes(redis_obj, owner_doc_key)
    kept = [item for item in episodes if _as_text(item.get("id")).strip() not in targets]
    deleted = len(episodes) - len(kept)
    if deleted:
        _memory_save_json(
            redis_obj,
            _memory_aux_key("episodes", owner_doc_key),
            {
                "schema_version": _MEMORY_AUX_SCHEMA_VERSION,
                "owner_doc_key": owner_doc_key,
                "updated_at": time.time(),
                "episodes": kept,
            },
        )
    return deleted


def _memory_open_loops(episodes: List[Dict[str, Any]], *, max_items: int = 5) -> List[Dict[str, Any]]:
    rows = [
        item for item in episodes
        if item.get("status") == "open" and (item.get("kind") in {"project", "commitment"} or item.get("commitments"))
    ]
    rows.sort(key=lambda item: (-float(item.get("salience") or 0.0), -float(item.get("occurred_at") or 0.0)))
    return rows[: max(1, int(max_items))]


def _memory_is_sensitive_key(key: Any) -> bool:
    token = normalize_fact_key(key)
    return any(part in token for part in _SENSITIVE_FACT_PARTS)


def _memory_profile_card(redis_obj: Any, owner_doc_key: str, doc: Dict[str, Any], *, max_items: int) -> Dict[str, Any]:
    cache_key = _memory_aux_key("profile", owner_doc_key)
    source_updated_at = _as_float(doc.get("last_updated"), 0.0, min_value=0.0)
    cached = _memory_load_json(redis_obj, cache_key, {})
    if _as_float(cached.get("source_updated_at"), -1.0) == source_updated_at and isinstance(cached.get("items"), list):
        return cached
    facts = doc.get("facts") if isinstance(doc.get("facts"), dict) else {}
    ranked: List[Tuple[int, float, float, str, Dict[str, Any]]] = []
    for key, fact in facts.items():
        if not isinstance(fact, dict) or _memory_is_sensitive_key(key):
            continue
        priority = _PROFILE_CARD_PRIORITY.index(key) if key in _PROFILE_CARD_PRIORITY else len(_PROFILE_CARD_PRIORITY) + 1
        ranked.append(
            (
                priority,
                -_as_float(fact.get("confidence"), 0.0, min_value=0.0, max_value=1.0),
                -_as_float(fact.get("last_confirmed_at"), _as_float(fact.get("updated_at"), 0.0), min_value=0.0),
                key,
                fact,
            )
        )
    ranked.sort(key=lambda row: row[:4])
    items = [
        {"key": key, "value": fact.get("value")}
        for _priority, _confidence, _confirmed, key, fact in ranked[: max(3, int(max_items))]
    ]
    card = {
        "schema_version": _MEMORY_AUX_SCHEMA_VERSION,
        "owner_doc_key": owner_doc_key,
        "source_updated_at": source_updated_at,
        "updated_at": time.time(),
        "items": items,
    }
    _memory_save_json(redis_obj, cache_key, card)
    return card


def _memory_consolidate_doc(doc: Dict[str, Any], *, scope: str, now_ts: Optional[float] = None) -> int:
    ts_now = float(now_ts if now_ts is not None else time.time())
    facts = doc.get("facts") if isinstance(doc.get("facts"), dict) else {}
    changed = 0
    canonicalizer = _canonical_user_fact_key if scope == "user" else _canonical_room_fact_key
    for old_key in list(facts.keys()):
        canonical_key = canonicalizer(old_key)
        if not canonical_key or canonical_key == old_key:
            continue
        old_fact = facts.get(old_key)
        if not isinstance(old_fact, dict):
            continue
        current = facts.get(canonical_key)
        if not isinstance(current, dict):
            facts[canonical_key] = old_fact
            facts.pop(old_key, None)
            changed += 1
            continue
        same_value = _value_fingerprint(current.get("value")) == _value_fingerprint(old_fact.get("value"))
        if same_value:
            current["confidence"] = max(_coerce_confidence(current.get("confidence")), _coerce_confidence(old_fact.get("confidence")))
            current["evidence"] = _memory_string_list(
                list(_coerce_evidence(current.get("evidence"))) + list(_coerce_evidence(old_fact.get("evidence"))),
                max_items=12,
                max_chars=160,
            )
            current["last_confirmed_at"] = max(
                _as_float(current.get("last_confirmed_at"), 0.0),
                _as_float(old_fact.get("last_confirmed_at"), 0.0),
            )
            facts.pop(old_key, None)
            changed += 1
            continue
        current_updated = _as_float(current.get("updated_at"), 0.0)
        old_updated = _as_float(old_fact.get("updated_at"), 0.0)
        winner = old_fact if old_updated > current_updated else current
        loser = current if winner is old_fact else old_fact
        retired = doc.get("retired_facts") if isinstance(doc.get("retired_facts"), list) else []
        retired.append(
            {
                "key": canonical_key,
                "value": loser.get("value"),
                "confidence": _coerce_confidence(loser.get("confidence")),
                "evidence": _coerce_evidence(loser.get("evidence")),
                "valid_from": _as_float(loser.get("valid_from"), _as_float(loser.get("updated_at"), ts_now)),
                "valid_to": ts_now,
                "reason": "consolidated",
            }
        )
        doc["retired_facts"] = retired[-100:]
        facts[canonical_key] = winner
        facts.pop(old_key, None)
        changed += 1
    if changed:
        doc["last_updated"] = ts_now
    return changed


def _run_memory_consolidation(redis_obj: Any, settings: Dict[str, Any], *, now_ts: Optional[float] = None) -> Dict[str, int]:
    ts_now = float(now_ts if now_ts is not None else time.time())
    changed_docs = 0
    changed_facts = 0
    profile_cards = 0
    for pattern, scope in (("memory_plus_core:user:*", "user"), ("memory_plus_core:room:*", "room")):
        try:
            keys = list(redis_obj.scan_iter(match=pattern, count=300))
        except Exception:
            keys = []
        for raw_key in keys:
            key = _as_text(raw_key).strip()
            if not key:
                continue
            try:
                doc = load_doc(redis_obj, key, now=ts_now)
                changes = _memory_consolidate_doc(doc, scope=scope, now_ts=ts_now)
                if changes:
                    save_doc(redis_obj, key, doc, now=ts_now)
                    changed_docs += 1
                    changed_facts += changes
                if scope == "user":
                    _memory_profile_card(
                        redis_obj,
                        key,
                        doc,
                        max_items=_as_int(settings.get("profile_card_items"), 8, min_value=3, max_value=20),
                    )
                    profile_cards += 1
            except Exception:
                logger.exception("[memory_plus_core] consolidation failed for %s", key)
    return {"consolidated_docs": changed_docs, "consolidated_facts": changed_facts, "profile_cards": profile_cards}


def _memory_retrieval_feedback(redis_obj: Any, owner_doc_key: str, result_ids: List[str]) -> None:
    if not result_ids:
        return
    key = _memory_aux_key("retrieval", owner_doc_key)
    payload = _memory_load_json(redis_obj, key, {"counts": {}})
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    now_ts = time.time()
    for result_id in result_ids:
        token = _memory_clean_text(result_id, max_chars=160)
        if not token:
            continue
        row = counts.get(token) if isinstance(counts.get(token), dict) else {}
        counts[token] = {
            "count": min(100000, _as_int(row.get("count"), 0, min_value=0) + 1),
            "last_retrieved_at": now_ts,
        }
    payload.update({"schema_version": _MEMORY_AUX_SCHEMA_VERSION, "counts": counts, "updated_at": now_ts})
    _memory_save_json(redis_obj, key, payload)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _as_int(value: Any, default: int, min_value: int = 0, max_value: Optional[int] = None) -> int:
    try:
        out = int(float(value))
    except Exception:
        out = int(default)
    if out < min_value:
        out = min_value
    if max_value is not None and out > max_value:
        out = max_value
    return out


def _as_float(value: Any, default: float, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < min_value:
        out = min_value
    if max_value is not None and out > max_value:
        out = max_value
    return out


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = _as_text(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


_PEOPLE_API_MODULE: Any = None
_PEOPLE_API_UNAVAILABLE = False


def _people_api_module() -> Any:
    global _PEOPLE_API_MODULE, _PEOPLE_API_UNAVAILABLE
    if _PEOPLE_API_MODULE is not None:
        return _PEOPLE_API_MODULE
    if _PEOPLE_API_UNAVAILABLE:
        return None

    try:
        import people as people_module  # type: ignore

        _PEOPLE_API_MODULE = people_module
        return _PEOPLE_API_MODULE
    except Exception:
        pass

    try:
        candidate = Path(__file__).resolve().parents[2] / "Tater" / "people.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("tater_people_api", candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _PEOPLE_API_MODULE = module
                return _PEOPLE_API_MODULE
    except Exception:
        pass

    _PEOPLE_API_UNAVAILABLE = True
    return None


def _people_person_rows(redis_obj: Any) -> List[Dict[str, Any]]:
    module = _people_api_module()
    load_store = getattr(module, "load_store", None) if module is not None else None
    if not callable(load_store):
        return []
    try:
        store = load_store(redis_obj)
    except Exception:
        return []
    people = list(store.get("people") or []) if isinstance(store, dict) else []
    rows = [dict(row) for row in people if isinstance(row, dict)]
    rows.sort(key=lambda row: (_as_text(row.get("display_name")).strip().lower(), _as_text(row.get("id")).strip()))
    return rows


def _people_person_name(person_id: Any, redis_obj: Any) -> str:
    wanted = _as_text(person_id).strip()
    if not wanted:
        return ""
    for person in _people_person_rows(redis_obj):
        if _as_text(person.get("id")).strip() == wanted:
            return _as_text(person.get("display_name")).strip()
    return ""


def _people_person_options(redis_obj: Any) -> List[Dict[str, str]]:
    options = [{"value": "", "label": "Choose a person"}]
    for person in _people_person_rows(redis_obj):
        person_id = _as_text(person.get("id")).strip()
        display_name = _as_text(person.get("display_name")).strip()
        if not person_id or not display_name:
            continue
        options.append({"value": person_id, "label": display_name})
    return options


def _people_origin_alias_identity(origin: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if not isinstance(origin, dict):
        return "", ""
    resolution = origin.get("people_resolution")
    if isinstance(resolution, dict) and _as_text(resolution.get("match_type")).strip() != "alias":
        return "", ""
    person_id = _as_text(origin.get("master_user_id") or origin.get("person_id")).strip()
    if not person_id:
        return "", ""
    return person_id, _as_text(origin.get("person_name") or origin.get("display_name")).strip()


def _people_alias_resolution(
    *,
    redis_obj: Any,
    platform: Any,
    user_id: Any,
    display_name: Any = "",
    origin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    platform_name = normalize_segment(platform, default="")
    user_text = _as_text(user_id).strip()
    if not platform_name or not user_text:
        return {}

    module = _people_api_module()
    resolve_person = getattr(module, "resolve_person", None) if module is not None else None
    if not callable(resolve_person):
        return {}

    label = _as_text(display_name).strip() or user_text
    origin_payload = dict(origin) if isinstance(origin, dict) else {}
    origin_payload.setdefault("platform", platform_name)
    origin_payload.setdefault("user_id", user_text)
    origin_payload.setdefault("username", label)
    origin_payload.setdefault("display_name", label)
    if platform_name == "voice_core":
        origin_payload.setdefault("speaker_id", user_text)
        origin_payload.setdefault("speaker_name", label)

    try:
        resolved = resolve_person(platform=platform_name, origin=origin_payload, redis_client=redis_obj)
    except Exception:
        return {}
    if not isinstance(resolved, dict):
        return {}
    if not bool(resolved.get("matched")) or _as_text(resolved.get("match_type")).strip() != "alias":
        return {}

    person_id = _as_text(resolved.get("master_user_id") or resolved.get("person_id")).strip()
    if not person_id:
        return {}
    return {
        "person_id": person_id,
        "person_name": _as_text(resolved.get("display_name")).strip() or person_id,
        "resolved": resolved,
    }


def _memory_user_doc_target(
    *,
    redis_obj: Any,
    platform: Any,
    user_id: Any,
    display_name: Any = "",
    origin: Optional[Dict[str, Any]] = None,
    use_people_identities: bool = True,
) -> Dict[str, str]:
    platform_name = normalize_segment(platform, default="webui")
    user_text = _as_text(user_id).strip()
    legacy_key = user_doc_key(platform_name, user_text)
    if not user_text:
        return {"doc_key": "", "legacy_key": "", "person_id": "", "person_name": ""}

    if bool(use_people_identities):
        person_id, person_name = _people_origin_alias_identity(origin)
        if not person_id:
            resolved = _people_alias_resolution(
                redis_obj=redis_obj,
                platform=platform_name,
                user_id=user_text,
                display_name=display_name,
                origin=origin,
            )
            person_id = _as_text(resolved.get("person_id")).strip()
            person_name = _as_text(resolved.get("person_name")).strip()
        if person_id:
            return {
                "doc_key": identity_doc_key(person_id),
                "legacy_key": legacy_key,
                "person_id": person_id,
                "person_name": person_name or person_id,
            }

    return {"doc_key": legacy_key, "legacy_key": legacy_key, "person_id": "", "person_name": ""}


def _memory_plus_link_person(
    *,
    platform: Any,
    user_id: Any,
    person_id: Any,
    label: Any = "",
    merge_existing_doc: bool = True,
    delete_source_doc: bool = False,
) -> Dict[str, Any]:
    platform_name = normalize_segment(platform, default="")
    user_text = _as_text(user_id).strip()
    person_text = _as_text(person_id).strip()
    if not platform_name or not user_text:
        return {"ok": False, "error": "Invalid memory user target."}
    if not person_text:
        return {"ok": False, "error": "Choose a master person to link."}

    module = _people_api_module()
    attach_alias = getattr(module, "attach_alias", None) if module is not None else None
    if not callable(attach_alias):
        return {"ok": False, "error": "People API is not available."}

    alias_label = _as_text(label).strip() or _memory_plus_user_name(platform_name, user_text) or user_text
    try:
        attach_alias(
            person_id=person_text,
            platform=platform_name,
            external_id=user_text,
            label=alias_label,
            kind="memory_user",
            redis_client=redis_client,
        )
    except Exception as exc:
        return {"ok": False, "error": f"People link failed: {exc}"}

    _save_user_label(platform_name, user_text, alias_label)
    person_name = _people_person_name(person_text, redis_client) or person_text
    _save_user_label("identity", person_text, person_name)

    merged = 0
    deleted = 0
    source_key = user_doc_key(platform_name, user_text)
    target_key = identity_doc_key(person_text)
    if bool(merge_existing_doc) and source_key != target_key:
        try:
            source_doc = load_doc(redis_client, source_key, now=time.time())
            target_doc = load_doc(redis_client, target_key, now=time.time())
            merged = int(merge_doc_facts(target_doc, source_doc, min_confidence=0.0, now=time.time()) or 0)
            merged += _memory_store_episodes(
                redis_client,
                target_key,
                _memory_load_episodes(redis_client, source_key),
                min_salience=0.0,
                max_items=2000,
                now_ts=time.time(),
            )
            if merged > 0:
                save_doc(redis_client, target_key, target_doc, now=time.time())
            if bool(delete_source_doc):
                deleted = int(
                    redis_client.delete(
                        source_key,
                        _memory_aux_key("episodes", source_key),
                        _memory_aux_key("profile", source_key),
                        _memory_aux_key("retrieval", source_key),
                    )
                    or 0
                )
        except Exception as exc:
            return {"ok": False, "error": f"Linked People alias, but memory merge failed: {exc}"}

    return {
        "ok": True,
        "person_id": person_text,
        "person_name": person_name,
        "merged": merged,
        "deleted": deleted,
    }



def _resolve_enabled_platforms() -> List[str]:
    auto_enabled = ["webui", "voice_core"]
    for platform, state_key in _AUTO_STATE_KEYS.items():
        if _as_text(redis_client.get(state_key)).strip().lower() == "true":
            auto_enabled.append(platform)
    return auto_enabled


def _voice_history_context_key(conversation_id: Any) -> str:
    conv_id = _as_text(conversation_id).strip()
    if not conv_id:
        return ""
    return f"tater:voice:conv:{conv_id}:ctx"


def _load_voice_history_context(conversation_id: Any) -> Dict[str, Any]:
    key = _voice_history_context_key(conversation_id)
    if not key:
        return {}
    try:
        raw = redis_client.get(key)
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        parsed = json.loads(_as_text(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _voice_context_room_id(context: Dict[str, Any], fallback: Any) -> str:
    if isinstance(context, dict):
        for key in (
            "area_id",
            "area_name",
            "room_id",
            "room_name",
            "device_id",
            "satellite_selector",
        ):
            text = _as_text(context.get(key)).strip()
            if text:
                return text
    return _as_text(fallback).strip()


def _voice_context_room_name(context: Dict[str, Any]) -> str:
    if not isinstance(context, dict):
        return ""
    for key in (
        "area_name",
        "room_name",
        "device_name",
        "satellite_name",
        "device_id",
        "satellite_selector",
    ):
        text = _as_text(context.get(key)).strip()
        if text:
            return text
    return ""


def _voice_context_user(context: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(context, dict):
        return "", ""
    speaker_id = _as_text(context.get("speaker_id")).strip()
    speaker_name = _as_text(context.get("speaker_name")).strip()
    if speaker_id:
        return speaker_id, speaker_name or speaker_id
    if speaker_name:
        return speaker_name, speaker_name
    return "", ""


def _memory_scope_id_for_history(platform: str, history_scope_id: str) -> str:
    if normalize_segment(platform, default="") != "voice_core":
        return history_scope_id
    return _voice_context_room_id(_load_voice_history_context(history_scope_id), history_scope_id)


def _discover_scopes(platform: str) -> List[Tuple[str, str]]:
    p = normalize_segment(platform, default="")
    if not p:
        return []

    if p == "webui":
        try:
            if int(redis_client.llen("webui:chat_history") or 0) > 0:
                return [("webui:chat_history", "chat")]
        except Exception:
            return []
        return []

    specs: List[Tuple[str, str, str]] = []
    if p == "discord":
        # Primary keyspace used by discord_portal.save_message.
        specs.append(("tater:channel:*:history", "tater:channel:", ":history"))
        # Compatibility for older/alternate key layouts.
        specs.append(("tater:discord:channel:*:history", "tater:discord:channel:", ":history"))
        specs.append(("tater:discord:dm:*:history", "tater:discord:dm:", ":history"))
    else:
        spec = _HISTORY_SCAN_SPECS.get(p)
        if spec:
            specs.append(spec)
    if not specs:
        return []

    out: List[Tuple[str, str]] = []
    seen_keys: set[str] = set()
    try:
        for pattern, prefix, suffix in specs:
            for raw_key in redis_client.scan_iter(match=pattern, count=500):
                key = _as_text(raw_key).strip()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                if not key.startswith(prefix) or not key.endswith(suffix):
                    continue
                scope_id = key[len(prefix) : len(key) - len(suffix)].strip()
                if not scope_id:
                    continue
                out.append((key, scope_id))
    except Exception:
        return []

    out.sort(key=lambda row: (str(row[1]), str(row[0])))
    return out


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        marker = _as_text(content.get("marker")).strip().lower()
        if marker in _MARKER_SKIP:
            return ""
        if marker == "plugin_response":
            return _message_content_text(content.get("content"))
        if marker == "plugin_call":
            plugin_name = _as_text(content.get("plugin")).strip()
            return f"[tool_call:{plugin_name}]" if plugin_name else "[tool_call]"

        content_type = _as_text(content.get("type")).strip().lower()
        if content_type in {"image", "audio", "video", "file"}:
            name = _as_text(content.get("name")).strip()
            return f"[{content_type}:{name}]" if name else f"[{content_type}]"

        for key in ("text", "content", "message", "value"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except Exception:
            return _as_text(content).strip()
    if isinstance(content, list):
        parts = [_message_content_text(item) for item in content]
        joined = "\n".join([part for part in parts if part])
        return joined.strip()
    if content is None:
        return ""
    return _as_text(content).strip()


def _message_timestamp(entry: Dict[str, Any], now_ts: float) -> float:
    for key in ("timestamp", "ts", "time", "updated_at"):
        if key in entry:
            return _as_float(entry.get(key), now_ts, min_value=0.0)
    content = entry.get("content")
    if isinstance(content, dict):
        for key in ("timestamp", "ts", "time", "updated_at"):
            if key in content:
                return _as_float(content.get(key), now_ts, min_value=0.0)
    return now_ts


_MEMORY_CURSOR_VERSION = 2
_MEMORY_CURSOR_ANCHOR_ROWS = 4
_MEMORY_CURSOR_SCAN_MIN_ROWS = 512
_MEMORY_CURSOR_SCAN_MAX_ROWS = 2000


def _history_row_fingerprint(raw_entry: Any) -> str:
    raw_text = _as_text(raw_entry)
    return hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()[:32]


def _normalized_message_id(
    platform: str,
    scope_id: str,
    index: int,
    raw_entry: Any = None,
) -> str:
    p = normalize_segment(platform, default="platform")
    s = normalize_segment(scope_id, default="scope")
    if raw_entry is not None:
        return f"msg_{p}_{s}_{_history_row_fingerprint(raw_entry)}"
    return f"msg_{p}_{s}_{int(index)}"


def _parse_history_cursor(raw_cursor: Any) -> Dict[str, Any]:
    text = _as_text(raw_cursor).strip()
    if not text:
        return {"version": 0, "position": 0, "anchor": []}

    try:
        payload = json.loads(text)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        version = _as_int(payload.get("version") or payload.get("v"), 0, min_value=0)
        if version >= _MEMORY_CURSOR_VERSION:
            raw_anchor = payload.get("anchor")
            anchor = [
                _as_text(value).strip()
                for value in (raw_anchor if isinstance(raw_anchor, list) else [])
                if _as_text(value).strip()
            ][-_MEMORY_CURSOR_ANCHOR_ROWS:]
            return {
                "version": _MEMORY_CURSOR_VERSION,
                "position": _as_int(payload.get("position"), 0, min_value=0),
                "anchor": anchor,
            }

    # Memory Core versions through 1.0.29 stored only a numeric list index.
    return {
        "version": 1,
        "position": _as_int(raw_cursor, 0, min_value=0),
        "anchor": [],
    }


def _history_fingerprints(rows: Any) -> List[str]:
    return [_history_row_fingerprint(row) for row in (rows or [])]


def _history_cursor_start(
    *,
    history_key: str,
    total_len: int,
    cursor_state: Dict[str, Any],
    lookback_limit: int,
) -> int:
    version = _as_int(cursor_state.get("version"), 0, min_value=0)
    position = _as_int(cursor_state.get("position"), 0, min_value=0)

    if version <= 0:
        return 0

    if version < _MEMORY_CURSOR_VERSION:
        if position < total_len:
            return position
        # One-time bounded recovery for scopes that were already stuck at the
        # fixed length of a capped Redis history list.
        return max(0, total_len - lookback_limit)

    anchor = [
        _as_text(value).strip()
        for value in (cursor_state.get("anchor") or [])
        if _as_text(value).strip()
    ][-_MEMORY_CURSOR_ANCHOR_ROWS:]
    if not anchor:
        return 0 if position <= 0 else max(0, total_len - lookback_limit)

    anchor_len = len(anchor)
    if anchor_len <= position <= total_len:
        rows = redis_client.lrange(history_key, position - anchor_len, position - 1) or []
        if _history_fingerprints(rows) == anchor:
            return position

    # A capped list shifted. Find the previous anchor in a bounded tail and
    # resume immediately after it. Searching newest-to-oldest handles repeated
    # history rows conservatively.
    scan_count = min(
        total_len,
        max(
            _MEMORY_CURSOR_SCAN_MIN_ROWS,
            min(_MEMORY_CURSOR_SCAN_MAX_ROWS, lookback_limit * 4),
        ),
    )
    scan_start = max(0, total_len - scan_count)
    scan_rows = redis_client.lrange(history_key, scan_start, total_len - 1) or []
    scan_fingerprints = _history_fingerprints(scan_rows)
    last_start = len(scan_fingerprints) - anchor_len
    for offset in range(last_start, -1, -1):
        if scan_fingerprints[offset : offset + anchor_len] == anchor:
            return scan_start + offset + anchor_len

    # The list completely rolled over or was reset. The missing portion is no
    # longer recoverable, but the newest bounded window still is.
    return max(0, total_len - lookback_limit)


def _save_history_cursor(
    *,
    cursor_redis_key: str,
    history_key: str,
    position: int,
    total_len: int,
) -> None:
    safe_position = max(0, min(int(position), int(total_len)))
    anchor_start = max(0, safe_position - _MEMORY_CURSOR_ANCHOR_ROWS)
    anchor_rows = (
        redis_client.lrange(history_key, anchor_start, safe_position - 1) or []
        if safe_position > 0
        else []
    )
    payload = {
        "version": _MEMORY_CURSOR_VERSION,
        "position": safe_position,
        "list_length": max(0, int(total_len)),
        "anchor": _history_fingerprints(anchor_rows),
    }
    redis_client.set(
        cursor_redis_key,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )


def _normalize_history_entry(
    *,
    platform: str,
    scope_id: str,
    history_scope_id: Optional[str] = None,
    index: int,
    raw_entry: Any,
    now_ts: float,
) -> Optional[Dict[str, Any]]:
    try:
        entry = json.loads(_as_text(raw_entry))
    except Exception:
        return None
    if not isinstance(entry, dict):
        return None

    role_raw = _as_text(entry.get("role")).strip().lower()
    role = "user" if role_raw == "user" else "assistant"
    text = _message_content_text(entry.get("content"))
    if not text:
        return None

    source_scope_id = _as_text(history_scope_id or scope_id).strip()
    voice_context = _load_voice_history_context(source_scope_id) if platform == "voice_core" else {}
    voice_user_id, voice_user_name = _voice_context_user(voice_context)

    username = _as_text(entry.get("username")).strip()
    user_handle = _as_text(entry.get("user_handle")).strip()
    entry_user_id = _as_text(entry.get("user_id")).strip()
    if not username:
        username = entry_user_id or _as_text(entry.get("user")).strip()
    if platform == "voice_core" and role_raw == "user":
        if not entry_user_id and voice_user_id:
            entry_user_id = voice_user_id
        if not username and voice_user_name:
            username = voice_user_name

    if platform == "telegram":
        handle_text = user_handle.strip()
        if handle_text.startswith("@"):
            handle_text = handle_text[1:].strip()

        def _generic_telegram_name(value: str) -> bool:
            v = str(value or "").strip().lower()
            return (
                not v
                or v in {"telegram_user", "unknown_user", "unknown"}
                or v.startswith("telegram_user_")
            )

        preferred = ""
        if handle_text:
            preferred = handle_text
        elif not _generic_telegram_name(username):
            preferred = username
        elif entry_user_id and entry_user_id.lower() not in {"assistant", "unknown_user"}:
            # Stable fallback when Telegram display name is unavailable.
            preferred = f"tg_{entry_user_id}" if entry_user_id.lstrip("-").isdigit() else entry_user_id

        if preferred:
            username = preferred
    if role == "assistant":
        user_id = "assistant"
    elif entry_user_id and entry_user_id.lower() not in {"assistant", "unknown_user"}:
        # Prefer stable platform IDs when present; this avoids cross-user merges by display name.
        user_id = entry_user_id
    elif username:
        user_id = username
    elif platform in {"macos", "homeassistant", "homekit", "xbmc"}:
        user_id = scope_id
    else:
        user_id = "unknown_user"

    room_name = _extract_room_label_from_entry(entry, platform=platform, scope_id=scope_id)
    if platform == "voice_core" and not room_name:
        room_name = _voice_context_room_name(voice_context)

    return {
        "platform": platform,
        "room_id": scope_id,
        "room_name": room_name,
        "user_id": user_id,
        "user_name": username if username else user_id,
        "role": role,
        "timestamp": _message_timestamp(entry, now_ts),
        "text": text,
        "message_id": _normalized_message_id(
            platform,
            source_scope_id or scope_id,
            index,
            raw_entry,
        ),
    }


def _candidate_user_ids(messages: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        if _as_text(item.get("role")).strip().lower() != "user":
            continue
        user_id = _as_text(item.get("user_id")).strip()
        if not user_id or user_id.lower() in {"assistant", "unknown_user"}:
            continue
        if user_id not in out:
            out.append(user_id)
    return out


def _user_display_names(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in messages:
        if not isinstance(item, dict):
            continue
        if _as_text(item.get("role")).strip().lower() != "user":
            continue
        user_id = _as_text(item.get("user_id")).strip()
        if not user_id or user_id.lower() in {"assistant", "unknown_user"}:
            continue
        user_name = _as_text(item.get("user_name")).strip() or user_id
        if user_id not in out and user_name:
            out[user_id] = user_name
    return out


def _existing_user_profile_snapshot(
    platform: str,
    user_ids: List[str],
    *,
    now_ts: float,
    user_display_names: Optional[Dict[str, str]] = None,
    use_people_identities: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_user_id in user_ids:
        user_id = _as_text(raw_user_id).strip()
        if not user_id or user_id.lower() in {"assistant", "unknown_user"}:
            continue
        display_name = (
            _as_text((user_display_names or {}).get(user_id)).strip()
            if isinstance(user_display_names, dict)
            else ""
        )
        target = _memory_user_doc_target(
            redis_obj=redis_client,
            platform=platform,
            user_id=user_id,
            display_name=display_name or user_id,
            use_people_identities=use_people_identities,
        )
        doc_key = _as_text(target.get("doc_key")).strip() or user_doc_key(platform, user_id)
        try:
            doc = load_doc(redis_client, doc_key, now=now_ts)
        except Exception:
            doc = {}
        legacy_key = _as_text(target.get("legacy_key")).strip()
        if legacy_key and legacy_key != doc_key:
            try:
                legacy_doc = load_doc(redis_client, legacy_key, now=now_ts)
            except Exception:
                legacy_doc = {}
            # Include not-yet-migrated alias facts in the extraction snapshot so
            # corrections can retire them before they are copied forward.
            merge_doc_facts(doc, legacy_doc, min_confidence=0.0, now=now_ts)
        facts = doc.get("facts")
        if not isinstance(facts, dict):
            continue
        profile: Dict[str, Dict[str, Any]] = {}
        fact_keys = list(_USER_ALLOWED_FACT_KEYS)
        fact_keys.extend(
            sorted(
                fact_key
                for fact_key in facts
                if fact_key not in _USER_ALLOWED_FACT_KEY_SET
                and bool(_canonical_user_fact_key(fact_key))
            )
        )
        for fact_key in fact_keys:
            fact = facts.get(fact_key)
            if not isinstance(fact, dict) or "value" not in fact:
                continue
            profile[fact_key] = {
                "value": fact.get("value"),
                "confidence": _as_float(fact.get("confidence"), 0.0, min_value=0.0, max_value=1.0),
                "updated_at": _as_float(fact.get("updated_at"), 0.0, min_value=0.0),
            }
        if profile:
            out[user_id] = profile
    return out


def _existing_room_profile_snapshot(
    platform: str,
    scope_id: str,
    *,
    now_ts: float,
) -> Dict[str, Dict[str, Any]]:
    try:
        doc = load_doc(redis_client, room_doc_key(platform, scope_id), now=now_ts)
    except Exception:
        doc = {}
    facts = doc.get("facts")
    if not isinstance(facts, dict):
        return {}

    profile: Dict[str, Dict[str, Any]] = {}
    fact_keys = list(_ROOM_ALLOWED_FACT_KEYS)
    fact_keys.extend(
        sorted(
            fact_key
            for fact_key in facts
            if fact_key not in _ROOM_ALLOWED_FACT_KEY_SET
            and bool(_canonical_room_fact_key(fact_key))
        )
    )
    for fact_key in fact_keys:
        fact = facts.get(fact_key)
        if not isinstance(fact, dict) or "value" not in fact:
            continue
        profile[fact_key] = {
            "value": fact.get("value"),
            "confidence": _as_float(fact.get("confidence"), 0.0, min_value=0.0, max_value=1.0),
            "updated_at": _as_float(fact.get("updated_at"), 0.0, min_value=0.0),
        }
    return profile


def _normalize_observation(
    row: Dict[str, Any],
    *,
    scope: str,
    known_message_ids: set[str],
    fallback_evidence_id: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    raw_fact_key = row.get("candidate_key") or row.get("key")
    scope_name = str(scope or "").strip().lower()
    if scope_name == "user":
        fact_key = _canonical_user_fact_key(raw_fact_key)
    elif scope_name == "room":
        fact_key = _canonical_room_fact_key(raw_fact_key)
    else:
        fact_key = normalize_fact_key(raw_fact_key)
    if not fact_key:
        return None
    if _is_transient_fact_key(fact_key):
        return None
    if "value" not in row:
        return None

    confidence = _as_float(row.get("confidence"), 0.0, min_value=0.0, max_value=1.0)
    # Platform memory is durable by default: disable TTL-based expiration.
    ttl_sec = 0

    evidence_raw = row.get("evidence") or row.get("evidence_message_ids") or []
    evidence: List[str] = []
    if isinstance(evidence_raw, (list, tuple)):
        for item in evidence_raw:
            msg_id = _as_text(item).strip()
            if not msg_id:
                continue
            if msg_id in known_message_ids and msg_id not in evidence:
                evidence.append(msg_id)
    elif isinstance(evidence_raw, str):
        msg_id = evidence_raw.strip()
        if msg_id in known_message_ids:
            evidence.append(msg_id)

    if not evidence and fallback_evidence_id:
        evidence = [fallback_evidence_id]

    return {
        "candidate_key": fact_key,
        "value": row.get("value"),
        "confidence": confidence,
        "ttl_sec": ttl_sec,
        "evidence": evidence[:8],
        "explicit_correction": bool(row.get("explicit_correction")),
        "operation": "replace" if bool(row.get("explicit_correction")) else "set",
    }


def _normalize_removal(
    row: Dict[str, Any],
    *,
    scope: str,
    known_message_ids: set[str],
    fallback_evidence_id: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    raw_fact_key = row.get("candidate_key") or row.get("key")
    scope_name = str(scope or "").strip().lower()
    if scope_name == "user":
        fact_key = _canonical_user_fact_key(raw_fact_key)
    elif scope_name == "room":
        fact_key = _canonical_room_fact_key(raw_fact_key)
    else:
        fact_key = normalize_fact_key(raw_fact_key)
    if not fact_key or _is_transient_fact_key(fact_key):
        return None

    confidence = _as_float(row.get("confidence"), 0.0, min_value=0.0, max_value=1.0)
    evidence_raw = row.get("evidence") or row.get("evidence_message_ids") or []
    evidence: List[str] = []
    if isinstance(evidence_raw, (list, tuple)):
        for item in evidence_raw:
            msg_id = _as_text(item).strip()
            if msg_id and msg_id in known_message_ids and msg_id not in evidence:
                evidence.append(msg_id)
    elif isinstance(evidence_raw, str):
        msg_id = evidence_raw.strip()
        if msg_id in known_message_ids:
            evidence.append(msg_id)
    if not evidence and fallback_evidence_id:
        evidence = [fallback_evidence_id]

    return {
        "candidate_key": fact_key,
        "confidence": confidence,
        "evidence": evidence[:8],
    }


def _normalize_episode_observation(
    row: Dict[str, Any],
    *,
    known_message_ids: set[str],
    message_timestamps: Dict[str, float],
    fallback_evidence_id: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    evidence = _memory_string_list(row.get("evidence") or row.get("evidence_message_ids"), max_items=8, max_chars=160)
    evidence = [item for item in evidence if item in known_message_ids]
    if not evidence and fallback_evidence_id:
        evidence = [fallback_evidence_id]
    occurred_at = max((message_timestamps.get(item, 0.0) for item in evidence), default=0.0)
    payload = dict(row)
    payload["evidence"] = evidence
    payload["occurred_at"] = occurred_at or time.time()
    return _memory_normalize_episode(payload)


def _normalize_observation_payload(
    parsed: Any,
    *,
    known_user_ids: List[str],
    known_message_ids: set[str],
    fallback_evidence_id: str,
    current_user_profiles: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    current_room_profile: Optional[Dict[str, Dict[str, Any]]] = None,
    message_timestamps: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    user_ids_lookup = {normalize_segment(user_id): user_id for user_id in known_user_ids}
    room_rows: List[Dict[str, Any]] = []
    user_rows: List[Dict[str, Any]] = []
    room_removals: List[Dict[str, Any]] = []
    user_removals: List[Dict[str, Any]] = []
    user_episodes: List[Dict[str, Any]] = []
    room_episodes: List[Dict[str, Any]] = []

    def _coerce_user_id(raw_value: Any) -> str:
        text = _as_text(raw_value).strip()
        if not text:
            return ""
        norm = normalize_segment(text)
        return user_ids_lookup.get(norm, text)

    def _append_room(row: Dict[str, Any]) -> None:
        obs = _normalize_observation(
            row,
            scope="room",
            known_message_ids=known_message_ids,
            fallback_evidence_id=fallback_evidence_id,
        )
        if obs:
            room_rows.append(obs)

    def _append_user(row: Dict[str, Any]) -> None:
        obs = _normalize_observation(
            row,
            scope="user",
            known_message_ids=known_message_ids,
            fallback_evidence_id=fallback_evidence_id,
        )
        if not obs:
            return
        user_id = _coerce_user_id(row.get("user_id"))
        if not user_id and len(known_user_ids) == 1:
            user_id = known_user_ids[0]
        if not user_id or user_id.lower() in {"assistant", "unknown_user"}:
            return
        obs["user_id"] = user_id
        user_rows.append(obs)

    def _append_room_removal(row: Dict[str, Any]) -> None:
        removal = _normalize_removal(
            row,
            scope="room",
            known_message_ids=known_message_ids,
            fallback_evidence_id=fallback_evidence_id,
        )
        if not removal:
            return
        fact_key = _as_text(removal.get("candidate_key")).strip()
        if fact_key not in (current_room_profile or {}):
            return
        room_removals.append(removal)

    def _append_user_removal(row: Dict[str, Any]) -> None:
        removal = _normalize_removal(
            row,
            scope="user",
            known_message_ids=known_message_ids,
            fallback_evidence_id=fallback_evidence_id,
        )
        if not removal:
            return
        user_id = _coerce_user_id(row.get("user_id"))
        if not user_id and len(known_user_ids) == 1:
            user_id = known_user_ids[0]
        if not user_id or user_id.lower() in {"assistant", "unknown_user"}:
            return
        fact_key = _as_text(removal.get("candidate_key")).strip()
        existing_profile = (current_user_profiles or {}).get(user_id) or {}
        if fact_key not in existing_profile:
            return
        removal["user_id"] = user_id
        user_removals.append(removal)

    def _append_episode(row: Dict[str, Any], *, user_scoped: bool) -> None:
        episode = _normalize_episode_observation(
            row,
            known_message_ids=known_message_ids,
            message_timestamps=message_timestamps or {},
            fallback_evidence_id=fallback_evidence_id,
        )
        if not episode:
            return
        if user_scoped:
            user_id = _coerce_user_id(row.get("user_id"))
            if not user_id and len(known_user_ids) == 1:
                user_id = known_user_ids[0]
            if not user_id or user_id.lower() in {"assistant", "unknown_user"}:
                return
            episode["user_id"] = user_id
            user_episodes.append(episode)
        else:
            room_episodes.append(episode)

    if isinstance(parsed, dict):
        user_list = parsed.get("user_observations") or parsed.get("user_updates") or []
        room_list = parsed.get("room_observations") or parsed.get("room_updates") or []
        user_removal_list = parsed.get("user_removals") or []
        room_removal_list = parsed.get("room_removals") or []
        user_episode_list = parsed.get("user_episodes") or []
        room_episode_list = parsed.get("room_episodes") or []
        if isinstance(user_list, list):
            for item in user_list:
                if isinstance(item, dict):
                    _append_user(item)
        if isinstance(room_list, list):
            for item in room_list:
                if isinstance(item, dict):
                    _append_room(item)
        if isinstance(user_removal_list, list):
            for item in user_removal_list:
                if isinstance(item, dict):
                    _append_user_removal(item)
        if isinstance(room_removal_list, list):
            for item in room_removal_list:
                if isinstance(item, dict):
                    _append_room_removal(item)
        if isinstance(user_episode_list, list):
            for item in user_episode_list:
                if isinstance(item, dict):
                    _append_episode(item, user_scoped=True)
        if isinstance(room_episode_list, list):
            for item in room_episode_list:
                if isinstance(item, dict):
                    _append_episode(item, user_scoped=False)

        generic_list = parsed.get("observations")
        if isinstance(generic_list, list):
            for item in generic_list:
                if not isinstance(item, dict):
                    continue
                scope = _as_text(item.get("scope")).strip().lower()
                if scope == "user" or item.get("user_id"):
                    _append_user(item)
                else:
                    _append_room(item)
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            scope = _as_text(item.get("scope")).strip().lower()
            if scope == "user" or item.get("user_id"):
                _append_user(item)
            else:
                _append_room(item)

    return {
        "user": user_rows,
        "room": room_rows,
        "user_removals": user_removals,
        "room_removals": room_removals,
        "user_episodes": user_episodes,
        "room_episodes": room_episodes,
    }


def _llm_extract_observations(
    llm_client: Any,
    *,
    platform: str,
    scope_id: str,
    messages: List[Dict[str, Any]],
    current_user_profiles: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    current_room_profile: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if llm_client is None:
        return None

    known_user_ids = _candidate_user_ids(messages)
    known_message_ids = {item.get("message_id") for item in messages if isinstance(item, dict) and item.get("message_id")}
    fallback_evidence_id = ""
    if messages:
        fallback_evidence_id = _as_text(messages[-1].get("message_id")).strip()

    payload = {
        "platform": platform,
        "room_id": scope_id,
        "known_user_ids": known_user_ids,
        "allowed_user_keys": list(_USER_ALLOWED_FACT_KEYS),
        "allowed_room_keys": list(_ROOM_ALLOWED_FACT_KEYS),
        "current_user_profiles": current_user_profiles or {},
        "current_room_profile": current_room_profile or {},
        "messages": messages,
    }
    system_prompt = (
        "Extract durable memory observations from normalized chat messages as incremental profile updates.\n"
        "Output strict JSON only with this shape:\n"
        "{"
        "\"user_observations\":[{\"user_id\":\"...\",\"candidate_key\":\"snake_case\",\"value\":...,\"confidence\":0-1,\"evidence\":[\"message_id\"]}],"
        "\"room_observations\":[{\"candidate_key\":\"snake_case\",\"value\":...,\"confidence\":0-1,\"evidence\":[\"message_id\"]}],"
        "\"user_removals\":[{\"user_id\":\"...\",\"candidate_key\":\"exact_existing_key\",\"confidence\":0-1,\"evidence\":[\"message_id\"]}],"
        "\"room_removals\":[{\"candidate_key\":\"exact_existing_key\",\"confidence\":0-1,\"evidence\":[\"message_id\"]}],"
        "\"user_episodes\":[{\"user_id\":\"...\",\"summary\":\"...\",\"kind\":\"decision|outcome|milestone|project|commitment|experience\",\"topics\":[\"...\"],\"outcome\":\"...\",\"commitments\":[\"...\"],\"status\":\"open|resolved|completed|cancelled|historical\",\"salience\":0-1,\"evidence\":[\"message_id\"]}],"
        "\"room_episodes\":[]"
        "}\n"
        "User profile schema and allowed candidate_key values (exact):\n"
        f"{_USER_SCHEMA_PROMPT_TEXT}\n"
        f"Flat allowed keys: {_USER_ALLOWED_FACT_KEYS_PROMPT}\n"
        "Room profile schema and allowed candidate_key values (exact):\n"
        f"{_ROOM_SCHEMA_PROMPT_TEXT}\n"
        f"Flat allowed keys: {_ROOM_ALLOWED_FACT_KEYS_PROMPT}\n"
        "The payload includes current_user_profiles and current_room_profile (already stored facts).\n"
        "Rules:\n"
        "- Use only explicit evidence from messages.\n"
        "- Keep keys short snake_case.\n"
        "- Prefer exact allowed user keys and room keys whenever one fits.\n"
        "- If no allowed key fits a durable, explicit, useful fact, create a concise snake_case key.\n"
        "- Dynamic user key examples: has_children, children_count, pet_names, favorite_console.\n"
        "- Dynamic room key examples: preferred_meeting_time, release_branch, build_server_name.\n"
        "- Do not create dynamic keys for one-off requests, current tasks, latest messages, credentials, secrets, tokens, passwords, or API keys.\n"
        "- Prefer concrete values (string/number/list) over booleans when possible.\n"
        "- Never guess sensitive profile attributes; include only directly stated details.\n"
        "- User observations must target one known_user_id.\n"
        "- Compare messages with current_user_profiles and emit only meaningful updates/additions.\n"
        "- For existing facts: if new evidence refines value or increases confidence, emit an update for the same key.\n"
        "- Set explicit_correction=true when the user directly changes or corrects an existing value; the newest direct statement must supersede the prior value even at equal confidence.\n"
        "- If a user explicitly says an existing fact is no longer true, has ended, was corrected away, or a condition is gone/recovered, emit a user_removals item using that exact existing key.\n"
        "- Example: when current_user_profiles contains a cough fact and the user says 'my cough is gone now', remove the existing cough key; do not store a new 'cough is gone' fact.\n"
        "- Never infer removals from silence, lack of recent mention, or ambiguous language.\n"
        "- Removal candidate_key values must exactly match keys currently present in the corresponding current profile.\n"
        "- If one stored value mixes resolved and still-valid details, emit an observation for the same key containing only the still-valid details instead of removing the whole key.\n"
        "- If no meaningful change for a user key, omit it.\n"
        "- Room observations are shared room context.\n"
        "- Compare messages with current_room_profile and emit only meaningful room updates/additions.\n"
        "- Use room_removals only when messages explicitly revoke or resolve an existing room fact.\n"
        "- If no meaningful change for a room key, omit it.\n"
        "- Do not include ttl/expiry fields.\n"
        "- Episodes are meaningful dated experiences, decisions, outcomes, milestones, ongoing projects, or commitments—not ordinary small talk or one-off requests.\n"
        "- Emit only high-value episodes that would help a future conversation. Use status=open for unfinished projects/commitments and historical for completed past context unless a more specific status applies.\n"
        "- Do not put secrets, credentials, or inferred sensitive traits in episodes.\n"
        "- If no valid observations exist, return empty arrays.\n"
    )
    native_system_prompt = (
        "Extract durable memory observations from the supplied normalized chat messages as incremental profile updates. "
        "Use only explicit message evidence. Prefer the supplied allowed user and room keys, creating a concise snake_case "
        "key only for a durable, explicit, useful fact that does not fit. Never store one-off requests, current tasks, "
        "credentials, secrets, tokens, passwords, API keys, or guessed sensitive attributes. Compare against the current "
        "profiles and emit only meaningful additions, refinements, or explicit corrections. Use removals only when a message "
        "explicitly revokes or resolves an existing fact, and copy the exact existing key. Evidence entries must be supplied "
        "message IDs; user observations must target a supplied known user ID."
    )
    observation_properties = {
        "candidate_key": {"type": "string"},
        "value": {},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "explicit_correction": {"type": "boolean"},
    }
    removal_properties = {
        "candidate_key": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    }
    episode_properties = {
        "summary": {"type": "string"},
        "kind": {"type": "string", "enum": sorted(_EPISODE_KINDS)},
        "topics": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string"},
        "commitments": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": sorted(_EPISODE_STATUSES)},
        "salience": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    }
    schema = {
        "type": "object",
        "properties": {
            "user_observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}, **observation_properties},
                    "required": ["user_id", "candidate_key", "value", "confidence", "evidence", "explicit_correction"],
                    "additionalProperties": False,
                },
            },
            "room_observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": observation_properties,
                    "required": ["candidate_key", "value", "confidence", "evidence", "explicit_correction"],
                    "additionalProperties": False,
                },
            },
            "user_removals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}, **removal_properties},
                    "required": ["user_id", "candidate_key", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            },
            "room_removals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": removal_properties,
                    "required": ["candidate_key", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            },
            "user_episodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}, **episode_properties},
                    "required": ["user_id", "summary", "kind", "topics", "outcome", "commitments", "status", "salience", "evidence"],
                    "additionalProperties": False,
                },
            },
            "room_episodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": episode_properties,
                    "required": ["summary", "kind", "topics", "outcome", "commitments", "status", "salience", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["user_observations", "room_observations", "user_removals", "room_removals", "user_episodes", "room_episodes"],
        "additionalProperties": False,
    }

    async def _call() -> Dict[str, Any]:
        return await llm_structured_chat(
            llm_client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            native_system_prompt=native_system_prompt,
            schema=schema,
            function_name="submit_memory_observations",
            max_tokens=None,
            temperature=0.1,
        )

    try:
        response = asyncio.run(_call())
    except Exception as exc:
        logger.warning("[memory_plus_core] LLM extraction failed for %s/%s: %s", platform, scope_id, exc)
        return None

    text = _as_text((response.get("message", {}) or {}).get("content")).strip()
    if not text:
        return {"user": [], "room": [], "user_removals": [], "room_removals": [], "user_episodes": [], "room_episodes": []}

    blob = extract_json(text) or text
    try:
        parsed = json.loads(blob)
    except Exception:
        logger.warning("[memory_plus_core] Non-JSON extraction response for %s/%s; skipping write.", platform, scope_id)
        return {"user": [], "room": [], "user_removals": [], "room_removals": [], "user_episodes": [], "room_episodes": []}

    return _normalize_observation_payload(
        parsed,
        known_user_ids=known_user_ids,
        known_message_ids=known_message_ids,
        fallback_evidence_id=fallback_evidence_id,
        current_user_profiles=current_user_profiles,
        current_room_profile=current_room_profile,
        message_timestamps={
            _as_text(item.get("message_id")).strip(): _as_float(item.get("timestamp"), time.time(), min_value=0.0)
            for item in messages
            if isinstance(item, dict) and _as_text(item.get("message_id")).strip()
        },
    )


def _process_scope(
    *,
    llm_client: Any,
    platform: str,
    history_key: str,
    scope_id: str,
    settings: Dict[str, Any],
) -> Dict[str, int]:
    lookback_limit = _as_int(settings.get("lookback_limit"), 80, min_value=10, max_value=500)
    min_confidence = _as_float(settings.get("min_confidence"), 0.65, min_value=0.0, max_value=1.0)
    write_user_memory = _as_bool(settings.get("write_user_memory"), True)
    write_room_memory = _as_bool(settings.get("write_room_memory"), True)
    use_people_identities = _as_bool(settings.get("use_people_identities"), True)
    episodic_memory = _as_bool(settings.get("episodic_memory"), True)
    episodic_min_salience = _as_float(settings.get("episodic_min_salience"), 0.7, min_value=0.0, max_value=1.0)
    episodic_max_items = _as_int(settings.get("episodic_max_items"), 250, min_value=25, max_value=2000)

    history_scope_id = _as_text(scope_id).strip()
    memory_scope_id = _memory_scope_id_for_history(platform, history_scope_id) or history_scope_id

    cursor_redis_key = cursor_key(platform, history_scope_id)
    raw_cursor = redis_client.get(cursor_redis_key)
    cursor_state = _parse_history_cursor(raw_cursor)

    total_len = _as_int(redis_client.llen(history_key), 0, min_value=0)
    if total_len <= 0:
        return {"processed_messages": 0, "updated_facts": 0, "updated_docs": 0}

    cursor_idx = _history_cursor_start(
        history_key=history_key,
        total_len=total_len,
        cursor_state=cursor_state,
        lookback_limit=lookback_limit,
    )

    if cursor_idx >= total_len:
        _save_history_cursor(
            cursor_redis_key=cursor_redis_key,
            history_key=history_key,
            position=total_len,
            total_len=total_len,
        )
        return {"processed_messages": 0, "updated_facts": 0, "updated_docs": 0}

    end_idx_exclusive = min(total_len, cursor_idx + lookback_limit)
    raw_rows = redis_client.lrange(history_key, cursor_idx, end_idx_exclusive - 1) or []
    if not raw_rows:
        return {"processed_messages": 0, "updated_facts": 0, "updated_docs": 0}

    ts_now = time.time()
    normalized_messages: List[Dict[str, Any]] = []
    for offset, raw_row in enumerate(raw_rows):
        abs_index = cursor_idx + offset
        item = _normalize_history_entry(
            platform=platform,
            scope_id=memory_scope_id,
            history_scope_id=history_scope_id,
            index=abs_index,
            raw_entry=raw_row,
            now_ts=ts_now,
        )
        if item:
            normalized_messages.append(item)

    if not normalized_messages:
        _save_history_cursor(
            cursor_redis_key=cursor_redis_key,
            history_key=history_key,
            position=end_idx_exclusive,
            total_len=total_len,
        )
        return {"processed_messages": 0, "updated_facts": 0, "updated_docs": 0}

    _remember_scope_labels(platform, memory_scope_id, normalized_messages)

    known_user_ids = _candidate_user_ids(normalized_messages)
    user_display_names = _user_display_names(normalized_messages)
    current_user_profiles = _existing_user_profile_snapshot(
        platform,
        known_user_ids,
        now_ts=ts_now,
        user_display_names=user_display_names,
        use_people_identities=use_people_identities,
    )
    current_room_profile = _existing_room_profile_snapshot(
        platform,
        memory_scope_id,
        now_ts=ts_now,
    )

    observations = _llm_extract_observations(
        llm_client,
        platform=platform,
        scope_id=memory_scope_id,
        messages=normalized_messages,
        current_user_profiles=current_user_profiles,
        current_room_profile=current_room_profile,
    )
    if observations is None:
        return {"processed_messages": 0, "updated_facts": 0, "updated_docs": 0}

    updated_facts = 0
    updated_docs = 0

    if write_room_memory:
        r_key = room_doc_key(platform, memory_scope_id)
        room_doc = load_doc(redis_client, r_key, now=ts_now)
        room_changed = False
        if _scrub_transient_facts(room_doc, now=ts_now) > 0:
            room_changed = True
        if _scrub_non_schema_room_facts(room_doc, now=ts_now) > 0:
            room_changed = True
        for obs in observations.get("room", []):
            if merge_observation(
                room_doc,
                obs,
                min_confidence=min_confidence,
                default_ttl_sec=0,
                allow_new_keys=True,
                now=ts_now,
            ):
                updated_facts += 1
                room_changed = True
        room_removal_keys = [
            _as_text(row.get("candidate_key")).strip()
            for row in observations.get("room_removals", [])
            if _as_float(row.get("confidence"), 0.0, min_value=0.0, max_value=1.0) >= min_confidence
        ]
        removed_room_facts = retire_fact_keys(room_doc, room_removal_keys, reason="ended", now=ts_now)
        if removed_room_facts > 0:
            updated_facts += removed_room_facts
            room_changed = True
        if room_changed:
            save_doc(redis_client, r_key, room_doc, now=ts_now)
            updated_docs += 1
        if episodic_memory:
            stored_room_episodes = _memory_store_episodes(
                redis_client,
                r_key,
                list(observations.get("room_episodes") or []),
                min_salience=episodic_min_salience,
                max_items=episodic_max_items,
                now_ts=ts_now,
            )
            if stored_room_episodes:
                updated_facts += stored_room_episodes
                updated_docs += 1

    if write_user_memory:
        by_user: Dict[str, List[Dict[str, Any]]] = {}
        for obs in observations.get("user", []):
            user_id = _as_text(obs.get("user_id")).strip()
            if not user_id:
                continue
            by_user.setdefault(user_id, []).append(obs)
        removals_by_user: Dict[str, List[Dict[str, Any]]] = {}
        for removal in observations.get("user_removals", []):
            user_id = _as_text(removal.get("user_id")).strip()
            if not user_id:
                continue
            removals_by_user.setdefault(user_id, []).append(removal)
        episodes_by_user: Dict[str, List[Dict[str, Any]]] = {}
        for episode in observations.get("user_episodes", []):
            user_id = _as_text(episode.get("user_id")).strip()
            if not user_id:
                continue
            episodes_by_user.setdefault(user_id, []).append(episode)

        user_ids_to_update = list(by_user)
        user_ids_to_update.extend(
            user_id for user_id in removals_by_user if user_id not in by_user
        )
        user_ids_to_update.extend(
            user_id for user_id in episodes_by_user if user_id not in user_ids_to_update
        )
        for user_id in user_ids_to_update:
            rows = by_user.get(user_id, [])
            display_name = _as_text(user_display_names.get(user_id)).strip() or user_id
            target = _memory_user_doc_target(
                redis_obj=redis_client,
                platform=platform,
                user_id=user_id,
                display_name=display_name,
                use_people_identities=use_people_identities,
            )
            legacy_key = _as_text(target.get("legacy_key")).strip() or user_doc_key(platform, user_id)
            u_key = _as_text(target.get("doc_key")).strip() or legacy_key
            identity_id = _as_text(target.get("person_id")).strip()
            identity_name = _as_text(target.get("person_name")).strip() or display_name
            if identity_id and identity_name:
                _save_user_label("identity", identity_id, identity_name)
            user_doc = load_doc(redis_client, u_key, now=ts_now)
            user_changed = False
            legacy_doc: Optional[Dict[str, Any]] = None
            legacy_changed = False
            if u_key != legacy_key:
                legacy_doc = load_doc(redis_client, legacy_key, now=ts_now)
                if merge_doc_facts(user_doc, legacy_doc, min_confidence=0.0, now=ts_now) > 0:
                    user_changed = True
            if _scrub_transient_facts(user_doc, now=ts_now) > 0:
                user_changed = True
            if _scrub_non_schema_user_facts(user_doc, now=ts_now) > 0:
                user_changed = True
            for obs in rows:
                if merge_observation(
                    user_doc,
                    obs,
                    min_confidence=min_confidence,
                    default_ttl_sec=0,
                    allow_new_keys=True,
                    now=ts_now,
                ):
                    updated_facts += 1
                    user_changed = True
            user_removal_keys = [
                _as_text(row.get("candidate_key")).strip()
                for row in removals_by_user.get(user_id, [])
                if _as_float(row.get("confidence"), 0.0, min_value=0.0, max_value=1.0) >= min_confidence
            ]
            removed_user_facts = retire_fact_keys(user_doc, user_removal_keys, reason="ended", now=ts_now)
            if removed_user_facts > 0:
                updated_facts += removed_user_facts
                user_changed = True
            if legacy_doc is not None and forget_fact_keys(legacy_doc, user_removal_keys) > 0:
                # Clear the alias copy too, otherwise the next migration pass
                # could resurrect a fact the user just resolved.
                legacy_changed = True
            if user_changed:
                save_doc(redis_client, u_key, user_doc, now=ts_now)
                updated_docs += 1
            if legacy_doc is not None and legacy_changed:
                save_doc(redis_client, legacy_key, legacy_doc, now=ts_now)
                updated_docs += 1
            if episodic_memory:
                stored_user_episodes = _memory_store_episodes(
                    redis_client,
                    u_key,
                    episodes_by_user.get(user_id, []),
                    min_salience=episodic_min_salience,
                    max_items=episodic_max_items,
                    now_ts=ts_now,
                )
                if stored_user_episodes:
                    updated_facts += stored_user_episodes
                    updated_docs += 1

    _save_history_cursor(
        cursor_redis_key=cursor_redis_key,
        history_key=history_key,
        position=end_idx_exclusive,
        total_len=total_len,
    )
    return {
        "processed_messages": len(normalized_messages),
        "updated_facts": updated_facts,
        "updated_docs": updated_docs,
    }


def _run_cycle(llm_client: Any, settings: Dict[str, Any]) -> Dict[str, int]:
    enabled_platforms = _resolve_enabled_platforms()
    enabled_with_history = 0
    processed_messages = 0
    updated_facts = 0
    updated_docs = 0
    scanned_scopes = 0

    for platform in enabled_platforms:
        scopes = _discover_scopes(platform)
        if scopes:
            enabled_with_history += 1
        for history_key, scope_id in scopes:
            scanned_scopes += 1
            result = _process_scope(
                llm_client=llm_client,
                platform=platform,
                history_key=history_key,
                scope_id=scope_id,
                settings=settings,
            )
            processed_messages += int(result.get("processed_messages") or 0)
            updated_facts += int(result.get("updated_facts") or 0)
            updated_docs += int(result.get("updated_docs") or 0)

    stats = {
        "enabled_platform_count": enabled_with_history,
        "scanned_scopes": scanned_scopes,
        "processed_messages": processed_messages,
        "updated_facts": updated_facts,
        "updated_docs": updated_docs,
    }
    return stats


def _run_memory_cycle_once(llm_client: Any, settings: Dict[str, Any]) -> Dict[str, Any]:
    global _MEMORY_TASK_STARTED_AT
    if not _MEMORY_TASK_LOCK.acquire(blocking=False):
        raise RuntimeError("Memory extraction is already running.")
    cycle_start = time.time()
    _MEMORY_TASK_STARTED_AT = cycle_start
    stats_key = "memory_plus_core:stats"
    try:
        stats = _run_cycle(llm_client, settings) if llm_client is not None else {
            "enabled_platform_count": 0,
            "scanned_scopes": 0,
            "processed_messages": 0,
            "updated_facts": 0,
            "updated_docs": 0,
        }
        try:
            last_consolidation = _as_float(redis_client.hget(stats_key, "last_consolidation_ts"), 0.0, min_value=0.0)
        except Exception:
            last_consolidation = 0.0
        consolidation_stats = {"consolidated_docs": 0, "consolidated_facts": 0, "profile_cards": 0}
        if cycle_start - last_consolidation >= 86_400:
            consolidation_stats = _run_memory_consolidation(redis_client, settings, now_ts=cycle_start)
            stats.update(consolidation_stats)
        finished_at = time.time()
        try:
            run_count = int(float(redis_client.hget(stats_key, "run_count") or 0)) + 1
        except Exception:
            run_count = 1
        redis_client.hset(
            stats_key,
            mapping={
                "last_run_ts": str(cycle_start),
                "last_finished_ts": str(finished_at),
                "last_duration_ms": str(max(0.0, (finished_at - cycle_start) * 1000.0)),
                "last_error": "",
                "run_count": str(run_count),
                "enabled_platform_count": str(stats.get("enabled_platform_count") or 0),
                "scanned_scopes": str(stats.get("scanned_scopes") or 0),
                "processed_messages": str(stats.get("processed_messages") or 0),
                "updated_facts": str(stats.get("updated_facts") or 0),
                "updated_docs": str(stats.get("updated_docs") or 0),
                "last_consolidation_ts": str(cycle_start if cycle_start - last_consolidation >= 86_400 else last_consolidation),
                "consolidated_docs": str(stats.get("consolidated_docs") or 0),
                "consolidated_facts": str(stats.get("consolidated_facts") or 0),
                "profile_cards": str(stats.get("profile_cards") or 0),
            },
        )
        return {"ok": True, **stats}
    except Exception as exc:
        finished_at = time.time()
        try:
            redis_client.hset(
                stats_key,
                mapping={
                    "last_finished_ts": str(finished_at),
                    "last_duration_ms": str(max(0.0, (finished_at - cycle_start) * 1000.0)),
                    "last_error": str(exc),
                },
            )
        except Exception:
            pass
        raise
    finally:
        _MEMORY_TASK_STARTED_AT = 0.0
        _MEMORY_TASK_LOCK.release()


def _sleep_with_stop(seconds: int, stop_event) -> None:
    target = max(1, int(seconds))
    elapsed = 0.0
    while elapsed < target:
        if stop_event and getattr(stop_event, "is_set", lambda: False)():
            return
        step = min(0.5, target - elapsed)
        time.sleep(step)
        elapsed += step


def _pending_memory_history_signature() -> Tuple[Tuple[str, int], ...]:
    pending: List[Tuple[str, int]] = []
    for platform in _resolve_enabled_platforms():
        for history_key, scope_id in _discover_scopes(platform):
            try:
                total_len = _as_int(redis_client.llen(history_key), 0, min_value=0)
                cursor = _parse_history_cursor(redis_client.get(cursor_key(platform, scope_id)))
                position = _as_int(cursor.get("position"), 0, min_value=0)
            except Exception:
                continue
            if total_len > position:
                pending.append((history_key, total_len))
    return tuple(sorted(pending))


def _wait_for_memory_work(
    *,
    interval_seconds: int,
    quiet_seconds: int,
    near_realtime: bool,
    stop_event: Any,
) -> None:
    if not near_realtime:
        _sleep_with_stop(interval_seconds, stop_event)
        return
    started = time.time()
    last_signature: Tuple[Tuple[str, int], ...] = ()
    last_change = started
    while time.time() - started < max(1, int(interval_seconds)):
        if stop_event and getattr(stop_event, "is_set", lambda: False)():
            return
        signature = _pending_memory_history_signature()
        now_ts = time.time()
        if signature != last_signature:
            last_signature = signature
            last_change = now_ts
        if signature and now_ts - last_change >= max(5, int(quiet_seconds)):
            return
        _sleep_with_stop(2, stop_event)


def run(stop_event=None):
    llm_client = None
    llm_ready_logged = False
    settings_logged = False
    logger.info("[memory_plus_core] started")

    while True:
        if stop_event and getattr(stop_event, "is_set", lambda: False)():
            break

        settings = _load_settings()
        interval_seconds = _as_int(settings.get("interval_seconds"), 180, min_value=30, max_value=3600)
        near_realtime = _as_bool(settings.get("near_realtime_extraction"), True)
        quiet_seconds = _as_int(settings.get("extraction_quiet_seconds"), 12, min_value=5, max_value=120)
        if not settings_logged:
            settings_logged = True
            logger.info("[memory_plus_core] extraction interval set to %ss", interval_seconds)
        if llm_client is None:
            try:
                llm_client = _get_primary_llm_client_from_env()
                if not llm_ready_logged and llm_client is not None:
                    llm_ready_logged = True
                    logger.info("[memory_plus_core] LLM client initialized")
            except Exception as exc:
                logger.warning("[memory_plus_core] failed to initialize LLM client: %s", exc)
                llm_client = None

        try:
            stats = _run_memory_cycle_once(llm_client, settings)
            if (
                int(stats.get("processed_messages") or 0) > 0
                or int(stats.get("updated_facts") or 0) > 0
                or int(stats.get("updated_docs") or 0) > 0
            ):
                logger.info(
                    "[memory_plus_core] cycle complete: scopes=%s processed=%s facts=%s docs=%s",
                    int(stats.get("scanned_scopes") or 0),
                    int(stats.get("processed_messages") or 0),
                    int(stats.get("updated_facts") or 0),
                    int(stats.get("updated_docs") or 0),
                )
        except Exception as exc:
            logger.exception("[memory_plus_core] cycle failed: %s", exc)

        _wait_for_memory_work(
            interval_seconds=interval_seconds,
            quiet_seconds=quiet_seconds,
            near_realtime=near_realtime,
            stop_event=stop_event,
        )

    logger.info("[memory_plus_core] stopped")


# ---- Embedded Memory UI (migrated from webui/webui_memory.py) ----

def _memory_plus_room_label_key(platform: Any, room_id: Any) -> str:
    platform_name = str(platform or "").strip().lower() or "unknown"
    scope_id = str(room_id or "").strip()
    if not scope_id:
        return ""
    return f"{_MEMORY_ROOM_LABEL_PREFIX}:{platform_name}:{scope_id}"


def _memory_plus_user_label_key(platform: Any, user_id: Any) -> str:
    platform_name = str(platform or "").strip().lower() or "unknown"
    scope_id = str(user_id or "").strip()
    if not scope_id:
        return ""
    return f"{_MEMORY_USER_LABEL_PREFIX}:{platform_name}:{scope_id}"


def _memory_plus_user_name(platform: Any, user_id: Any) -> str:
    platform_name = str(platform or "").strip().lower() or "unknown"
    scope_id = str(user_id or "").strip()
    if not scope_id:
        return "unknown"

    cache_key = (platform_name, scope_id)
    cached = _MEMORY_USER_NAME_CACHE.get(cache_key)
    if cached:
        return cached

    key = _memory_plus_user_label_key(platform_name, scope_id)
    if key:
        try:
            raw = redis_client.get(key)
        except Exception:
            raw = None
        label = str(raw or "").strip()
        if label:
            _MEMORY_USER_NAME_CACHE[cache_key] = label
            return label

    if platform_name == "identity":
        people_name = _people_person_name(scope_id, redis_client)
        if people_name:
            _MEMORY_USER_NAME_CACHE[cache_key] = people_name
            return people_name
        try:
            for raw_key in redis_client.scan_iter(match="memory_plus_core:identity_name:*", count=300):
                lookup_key = str(raw_key or "").strip()
                if not lookup_key:
                    continue
                mapped = str(redis_client.get(lookup_key) or "").strip()
                if mapped != scope_id:
                    continue
                identity_name = lookup_key.split("memory_plus_core:identity_name:", 1)[-1].strip()
                if identity_name:
                    _MEMORY_USER_NAME_CACHE[cache_key] = identity_name
                    return identity_name
        except Exception:
            pass

    history_match_specs = {
        "discord": "tater:channel:*:history",
        "telegram": "tater:telegram:*:history",
        "macos": "tater:macos:session:*:history",
        "homeassistant": "tater:ha:session:*:history",
        "homekit": "tater:homekit:session:*:history",
        "xbmc": "tater:xbmc:session:*:history",
        "meshtastic": "tater:meshtastic:*:history",
        "voice_core": "tater:voice:conv:*:history",
        "webui": "webui:chat_history",
    }
    history_pattern = history_match_specs.get(platform_name)
    if history_pattern:
        try:
            history_keys: List[str] = []
            if "*" in history_pattern:
                for raw_key in redis_client.scan_iter(match=history_pattern, count=200):
                    history_key = str(raw_key or "").strip()
                    if history_key:
                        history_keys.append(history_key)
            else:
                history_keys.append(history_pattern)

            for history_key in history_keys[:200]:
                rows = redis_client.lrange(history_key, -120, -1) or []
                for raw_row in reversed(rows):
                    try:
                        entry = json.loads(str(raw_row or ""))
                    except Exception:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    entry_user_id = str(entry.get("user_id") or "").strip()
                    if entry_user_id and entry_user_id != scope_id:
                        continue
                    username = str(entry.get("username") or entry.get("user") or "").strip()
                    if username and (entry_user_id == scope_id or username == scope_id):
                        _MEMORY_USER_NAME_CACHE[cache_key] = username
                        return username
        except Exception:
            pass

    _MEMORY_USER_NAME_CACHE[cache_key] = scope_id
    return scope_id


def _memory_plus_room_name(platform: Any, room_id: Any) -> str:
    platform_name = str(platform or "").strip().lower() or "unknown"
    scope_id = str(room_id or "").strip()
    if not scope_id:
        return "unknown"

    key = _memory_plus_room_label_key(platform_name, scope_id)
    if key:
        try:
            raw = redis_client.get(key)
        except Exception:
            raw = None
        label = str(raw or "").strip()
        if label:
            return label

    if platform_name == "telegram":
        try:
            candidates: List[str] = []
            for raw_name, raw_chat_id in redis_client.hscan_iter(_TELEGRAM_CHAT_LOOKUP_HASH, count=500):
                chat_id = str(raw_chat_id or "").strip()
                if chat_id != scope_id:
                    continue
                name = str(raw_name or "").strip()
                if not name:
                    continue
                if name.lstrip("-").isdigit():
                    continue
                candidates.append(name)
            if candidates:
                candidates.sort(
                    key=lambda value: (
                        value.lower().startswith("telegram_user"),
                        value.lower().startswith("unknown"),
                        len(value),
                        value.lower(),
                    )
                )
                return candidates[0]
        except Exception:
            pass

    return scope_id


def _memory_plus_room_display_name_from_row(row: Dict[str, Any]) -> str:
    room_id = str((row or {}).get("room_id") or "").strip() or "unknown"
    room_name = str((row or {}).get("room_name") or "").strip() or room_id
    return room_name


def _memory_plus_stats() -> Dict[str, Any]:
    raw = redis_client.hgetall("memory_plus_core:stats") or {}
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    stats = {
        "last_run_ts": _to_float(raw.get("last_run_ts"), 0.0),
        "last_finished_ts": _to_float(raw.get("last_finished_ts"), _to_float(raw.get("last_run_ts"), 0.0)),
        "last_duration_ms": _to_float(raw.get("last_duration_ms"), 0.0),
        "last_error": str(raw.get("last_error") or "").strip(),
        "run_count": _to_int(raw.get("run_count"), 0),
        "enabled_platform_count": _to_int(raw.get("enabled_platform_count"), 0),
        "scanned_scopes": _to_int(raw.get("scanned_scopes"), 0),
        "processed_messages": _to_int(raw.get("processed_messages"), 0),
        "updated_facts": _to_int(raw.get("updated_facts"), 0),
        "updated_docs": _to_int(raw.get("updated_docs"), 0),
    }
    stats["last_run_text"] = _format_unix_ts(stats.get("last_run_ts"))
    return stats

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
            if int(redis_client.delete(META_KEY) or 0) > 0:
                deleted += 1
        except Exception:
            logger.exception("[memory_plus_core] wipe failed for %s", META_KEY)
    return deleted
