"""Phase 1 tests: fact-merge correction rules (§7.4), doc store basics, hook contracts."""

import json

NOW = 1_800_000_000.0


# ---------- §7.4 merge rules ----------

def _merge(core, doc, value, conf, **obs_extra):
    obs = {"candidate_key": "favorite_food", "value": value, "confidence": conf}
    obs.update(obs_extra)
    changed = core.merge_observation(
        doc, obs, min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=NOW
    )
    return changed


def test_new_fact_stored(core):
    doc = core.empty_memory_doc(now=NOW)
    assert _merge(core, doc, "thai curry", 0.8) is True
    fact = doc["facts"]["favorite_food"]
    assert fact["value"] == "thai curry"
    assert fact["pending_alt"] is None
    assert fact["mentions"] == 1
    assert fact["first_seen"] == NOW


def test_same_value_bumps_mentions_and_evidence(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai curry", 0.8, evidence=["m1"])
    _merge(core, doc, "thai curry", 0.75, evidence=["m2"])
    fact = doc["facts"]["favorite_food"]
    assert fact["mentions"] == 2
    assert sorted(fact["evidence"]) == ["m1", "m2"]
    # pending alt cleared after a confirming repeat
    assert not fact.get("pending_alt")


def test_correction_not_blocked_by_confidence_ratchet(core):
    """The core fix: matured high-confidence fact is still correctable."""
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai curry", 0.9)
    _merge(core, doc, "thai curry", 0.9)  # ratchet to ~0.99
    assert doc["facts"]["favorite_food"]["value"] == "thai curry"
    merged = _merge(core, doc, "ramen", 0.7)
    assert merged is False  # first differing sighting goes to pending_alt
    fact = doc["facts"]["favorite_food"]
    assert fact["value"] == "thai curry"
    assert fact["pending_alt"]["value"] == "ramen"
    assert fact["pending_alt"]["count"] == 1


def test_pending_alt_applies_after_second_sighting(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai curry", 0.9)
    assert _merge(core, doc, "ramen", 0.7) is False
    assert _merge(core, doc, "ramen", 0.7) is True
    fact = doc["facts"]["favorite_food"]
    assert fact["value"] == "ramen"
    assert fact["previous"] == [{"value": "thai curry", "until": NOW}]
    assert fact["pending_alt"] is None


def test_supersedes_replaces_immediately(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai curry", 0.9)
    assert _merge(core, doc, "ramen", 0.6, supersedes=True) is True
    fact = doc["facts"]["favorite_food"]
    assert fact["value"] == "ramen"
    assert fact["previous"][0]["value"] == "thai curry"


def test_fresh_conf_close_enough_replaces(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai curry", 0.7)
    assert _merge(core, doc, "ramen", 0.62) is True  # 0.7 - 0.15 = 0.55
    fact = doc["facts"]["favorite_food"]
    assert fact["value"] == "ramen"
    assert fact["pending_alt"] is None


def test_stale_fact_replaced_by_age(core):
    doc = core.empty_memory_doc(now=NOW)
    obs = {"candidate_key": "favorite_food", "value": "thai curry", "confidence": 0.9}
    old = NOW - 40 * 86_400  # first_seen 40 days ago
    core.merge_observation(doc, obs, min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=old)
    merged = core.merge_observation(
        doc,
        {"candidate_key": "favorite_food", "value": "ramen", "confidence": 0.6},
        min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=NOW,
    )
    assert merged is True
    assert doc["facts"]["favorite_food"]["value"] == "ramen"


def test_conflicting_alternatives_toggle_but_no_flapping(core):
    """Two different alternates seen once each: no replace; latest alt pends."""
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai curry", 0.9)
    assert _merge(core, doc, "ramen", 0.6) is False
    assert _merge(core, doc, "pizza", 0.6) is False  # replaces the pending alt, no apply
    fact = doc["facts"]["favorite_food"]
    assert fact["value"] == "thai curry"
    assert fact["pending_alt"]["value"] == "pizza"
    assert fact["pending_alt"]["count"] == 1


def test_list_union_instead_of_replace(core):
    doc = core.empty_memory_doc(now=NOW)
    assert _merge(core, doc, ["thai"], 0.8) is True
    assert _merge(core, doc, ["sushi"], 0.7) is True
    assert sorted(doc["facts"]["favorite_food"]["value"]) == ["sushi", "thai"]


def test_list_supersedes_replaces(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, ["thai"], 0.8)
    assert _merge(core, doc, ["ramen"], 0.7, supersedes=True) is True
    assert doc["facts"]["favorite_food"]["value"] == ["ramen"]


def test_previous_capped_at_three(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "a", 0.9)
    for val in ("b", "c", "d", "e"):
        _merge(core, doc, val, 0.9, supersedes=True)
    prev = [p["value"] for p in doc["facts"]["favorite_food"]["previous"]]
    assert prev == ["b", "c", "d"]
    assert doc["facts"]["favorite_food"]["value"] == "e"


def test_below_min_confidence_ignored(core):
    doc = core.empty_memory_doc(now=NOW)
    assert _merge(core, doc, "thai curry", 0.3) is False
    assert doc["facts"] == {}


def test_merge_does_not_crash_on_garbage(core):
    doc = core.empty_memory_doc(now=NOW)
    assert core.merge_observation(doc, None, min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=NOW) is False
    assert core.merge_observation(doc, {"value": 1}, min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=NOW) is False


# ---------- doc store basics ----------

def test_doc_roundtrip(core, fake_redis_cls):
    r = fake_redis_cls()
    doc = core.empty_memory_doc(now=NOW)
    core.merge_observation(
        doc,
        {"candidate_key": "occupation", "value": "engineer", "confidence": 0.9, "category": "demographic_information"},
        min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=NOW,
    )
    core.save_doc(r, "memory_plus_core:user:identity:tester", doc, now=NOW)
    loaded = core.load_doc(r, "memory_plus_core:user:identity:tester", now=NOW)
    assert loaded["facts"]["occupation"]["value"] == "engineer"


def test_forget_fact_keys(core):
    doc = core.empty_memory_doc(now=NOW)
    _merge(core, doc, "thai", 0.8)
    _merge(core, doc, "rex", 0.9, candidate_key="pet_name")
    count = core.forget_fact_keys(doc, ["favorite_food"])
    assert count == 1
    assert "pet_name" in doc["facts"] and "favorite_food" not in doc["facts"]


# ---------- identity & namespaces ----------

def test_user_doc_key_uses_own_prefix(core):
    assert core.user_doc_key("macos", "u1") == "memory_plus_core:user:macos:u1"
    assert core.room_doc_key("discord", "chan1") == "memory_plus_core:room:discord:chan1"
    assert core.identity_doc_key("tester") == "memory_plus_core:user:identity:tester"


def test_resolve_identity_without_people(core, fake_redis_cls):
    r = fake_redis_cls()
    ident = core.resolve_identity_id(r, "webui", "user42", create=True, display_name="Tester")
    assert ident
    assert core.resolve_identity_id(r, "webui", "user42") == ident


def test_doc_keys_never_collide_with_stock(core):
    """Own namespace never emits stock mem:* keys."""
    for key in (
        core.user_doc_key("macos", "u1"),
        core.room_doc_key("discord", "chan1"),
        core.cursor_key("macos", "session1"),
    ):
        assert "mem:" != key[:4]
        assert key.startswith("memory_plus_core:")


# ---------- hook contracts (never raise) ----------

def test_hot_path_hooks_never_raise(core, fake_redis_cls):
    r = fake_redis_cls()
    garbage_origins = [None, {}, {"user_id": 42}, {"user_id": "u", "master_user_id": "p", "people_resolution": {"match_type": "alias"}}]
    for i, origin in enumerate(garbage_origins):
        payload = core.get_hydra_memory_context_payload(
            platform="webui" if i % 2 else None,
            scope=None,
            origin=origin,
            redis_client=r if i % 3 else None,
            user_text="hello there friend",
            request_text=None,
        )
        assert isinstance(payload, dict)
        for role in ("chat", "hermes"):
            frags = core.get_hydra_system_prompt_fragments(
                role=role, platform="webui", scope="", origin=origin,
                redis_client=r, memory_context=payload, user_text="hello", request_text="",
            )
            assert isinstance(frags, dict)


def test_run_hydra_kernel_tool_unknown_id(core):
    import asyncio

    async def go():
        return await core.run_hydra_kernel_tool(tool_id="not_our_tool", args={}, platform="webui")
    assert asyncio.run(go()) is None


def test_wipe_only_touches_own_namespace(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r  # wipe operates on the module-level redis client
    r.set("memory_plus_core:user:identity:x", "{}")
    r.set("memory_plus_core:meta", "{}")
    r.set("mem:user:keep:me", "{}")  # stock key must survive
    r.set("tater:something", "keep")
    assert core.wipe_memory_plus_core_data() >= 2
    assert r.exists("mem:user:keep:me") == 1
    assert r.exists("tater:something") == 1
    assert not r.keys("memory_plus_core:*") or r.keys("memory_plus_core:*") == []
