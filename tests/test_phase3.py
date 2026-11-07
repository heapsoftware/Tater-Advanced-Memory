"""Phase 3: consolidation upgrade (§7.6), initial backfill (§7.3), stock import (§7.8)."""

import json

NOW = 1_800_000_000.0


def _doc_with(core, *observations, now=NOW):
    doc = core.empty_memory_doc(now=now)
    for obs in observations:
        core.merge_observation(doc, obs, min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=now)
    return doc


def _make_fact(value="curry nights", confidence=0.9):
    return {
        "value": value,
        "confidence": confidence,
        "evidence": ["m1"],
        "valid_from": NOW,
        "updated_at": NOW,
        "first_seen": NOW,
        "mentions": 2,
        "source": "stated",
        "category": "preferences",
        "previous": [],
        "pending_alt": None,
    }


# ---------- §7.6 backups ----------

def test_backup_creates_snapshot_and_rotates(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    doc = _doc_with(core, {"candidate_key": "a", "value": 1, "confidence": 0.9})
    key = core.user_doc_key("webui", "user1")
    for day in range(5):
        ts = NOW + day * 86_400
        core._memory_backup_doc(r, key, doc, now_ts=ts)
    stamps = sorted(
        k.rsplit(":", 1)[-1]
        for k in r.keys("memory_plus_core:backup:*")
    )
    assert len(stamps) == 3, f"rotation kept {len(stamps)}: {stamps}"
    snap = json.loads(r.get(sorted(r.keys("memory_plus_core:backup:*"))[0]))
    assert snap["doc"]["facts"]["a"]["value"] == 1


# ---------- §7.6 validation (all-or-nothing) ----------

def _proposal(merges=(), rewrites=(), stale=(), profile_card=""):
    return {"merges": list(merges), "rewrites": list(rewrites), "stale": list(stale), "profile_card": profile_card}


def test_validation_rejects_missing_or_self_merges(core):
    doc = {"facts": {"pet_names": _make_fact()}}
    for proposal in (
        _proposal(merges=[{"from_key": "ghost", "to_key": "pets"}]),
        _proposal(merges=[{"from_key": "pet_names", "to_key": "pet_names"}]),
        _proposal(merges=[{"from_key": "", "to_key": "pets"}]),
    ):
        ok, reason, _ = core._memory_validate_consolidation_proposal(doc, proposal, profile_card_max_chars=600)
        assert not ok and reason


def test_validation_rejects_stale_flag_on_well_confirmed_fact(core):
    doc = {"facts": {"job": _make_fact(confidence=0.95, value="engineer")}}
    doc["facts"]["job"]["mentions"] = 5
    ok, reason, _ = core._memory_validate_consolidation_proposal(
        doc, _proposal(stale=[{"key": "job"}]), profile_card_max_chars=600
    )
    assert not ok and "well-confirmed" in reason


def test_validation_and_apply_roundtrip(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    doc = _doc_with(core, *[
        {"candidate_key": "pet_names", "value": "rex", "confidence": 0.85, "category": "family_background"},
        {"candidate_key": "weather_pref", "value": "cold", "confidence": 0.6},
    ])
    proposal = _proposal(
        merges=[{"from_key": "dog_name", "to_key": "pet_names", "reason": "dup"}],
        rewrites=[{"key": "weather_pref", "value": "cool nights"}],
        profile_card="Likes cool nights and has a dog named rex.",
    )
    # dog_name proposed but absent -> rejected wholesale
    proposal["merges"][0]["from_key"] = "dog_name"
    ok, reason, _ = core._memory_validate_consolidation_proposal(doc, proposal, profile_card_max_chars=600)
    assert not ok

    proposal["merges"][0]["from_key"] = "weather_pref_old"
    proposal["rewrites"] = [{"key": "missing_key", "value": "x"}]
    ok, _r2, _ = core._memory_validate_consolidation_proposal(doc, proposal, profile_card_max_chars=600)
    assert not ok

    # a fully valid proposal applies end to end
    doc2 = _doc_with(core, *[
        {"candidate_key": "pet_names", "value": "rex", "confidence": 0.85, "category": "family_background"},
        {"candidate_key": "dog_name", "value": "rex the labrador", "confidence": 0.7, "category": "family_background"},
        {"candidate_key": "weather_pref", "value": "cold", "confidence": 0.6},
    ])
    proposal = _proposal(
        merges=[{"from_key": "dog_name", "to_key": "pet_names", "reason": "duplicate"}],
        rewrites=[{"key": "weather_pref", "value": "cool nights"}],
        profile_card="Has a dog; prefers cool nights.",
    )
    ok, reason, validated = core._memory_validate_consolidation_proposal(doc2, proposal, profile_card_max_chars=600)
    assert ok, reason
    core._memory_apply_consolidation(doc2, validated, now_ts=NOW + 60)
    assert "dog_name" not in doc2["facts"]
    merged = doc2["facts"]["pet_names"]
    assert merged["value"] == "rex the labrador"  # newer value wins
    assert merged["previous"][0]["value"] == "rex"
    assert doc2["facts"]["weather_pref"]["value"] == "cool nights"
    card = doc2["profile_card"]
    assert card["source"] == "consolidation" and len(card["text"]) <= 600


def test_consolidation_run_rejected_proposal_touches_nothing(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r
    doc = _doc_with(core, {"candidate_key": "job", "value": "engineer", "confidence": 0.95})
    doc["facts"]["job"]["mentions"] = 5
    key = core.user_doc_key("webui", "user1")
    core.save_doc(r, key, doc, now=NOW)

    before = r.get(key)

    def fake_proposal(_llm, d, *, profile_card_max_chars):
        return _proposal(stale=[{"key": "job"}])

    monkeypatch.setattr(core, "_memory_consolidation_proposal", fake_proposal)
    summary = core._memory_plus_consolidation_run(r, "llm", {"profile_card_max_chars": 600}, now_ts=NOW)
    assert summary["rejected"] == 1
    assert summary["applied_docs"] == 0
    assert r.get(key) == before
    assert not r.keys("memory_plus_core:backup:*")

    dry = fake_redis_cls()
    monkeypatch.setattr(core, "_memory_consolidation_proposal", lambda *a, **k: _proposal(rewrites=[{"key": "job", "value": "x"}]))
    s2 = core._memory_plus_consolidation_run(dry, "llm", {}, now_ts=NOW, dry_run=True)
    assert s2.get("dry_run") is True
    assert s2["proposed"] == 0 and s2["docs"] == 0


def test_consolidation_run_applies_and_records(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r
    doc = _doc_with(core, {"candidate_key": "low_conf_note", "value": "stale stuff", "confidence": 0.6})
    key = core.user_doc_key("webui", "user1")
    core.save_doc(r, key, doc, now=NOW)

    monkeypatch.setattr(
        core,
        "_memory_consolidation_proposal",
        lambda *a, **k: _proposal(stale=[{"key": "low_conf_note"}], profile_card="Tidy person."),
    )
    summary = core._memory_plus_consolidation_run(r, "llm", {}, now_ts=NOW)
    assert summary["retired"] == 1
    after = core.load_doc(r, key, now=NOW)
    assert "low_conf_note" not in (after.get("facts") or {})
    assert after.get("retired_facts") and after["retired_facts"][-1]["reason"] == "consolidation_stale"


# ---------- §7.1 schema round-trip ----------

def test_fact_fields_survive_save_load_roundtrip(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    doc = _doc_with(core, {"candidate_key": "occupation", "value": "electrician", "confidence": 0.9, "category": "work"})
    doc["facts"]["occupation"]["mentions"] = 4
    doc["facts"]["occupation"]["source"] = "stated"
    doc["facts"]["occupation"]["previous"].append({"value": "plumber", "until": NOW - 10})
    doc["facts"]["occupation"]["pending_alt"] = {"value": "welder", "count": 1, "last_seen": NOW - 5}
    key = core.user_doc_key("webui", "user2")
    core.save_doc(r, key, doc, now=NOW)
    loaded = core.load_doc(r, key, now=NOW)
    fact = loaded["facts"]["occupation"]
    assert fact["mentions"] == 4 and fact["source"] == "stated"
    assert fact["category"] == "work"
    assert fact["previous"][0]["value"] == "plumber"
    assert fact["pending_alt"]["value"] == "welder" and fact["pending_alt"]["count"] == 1
    assert fact["first_seen"] == doc["facts"]["occupation"]["first_seen"]
    # second sighting of the alt value promotes it after reload
    core.merge_observation(loaded, {"candidate_key": "occupation", "value": "welder", "confidence": 0.9},
                           min_confidence=0.5, default_ttl_sec=0, allow_new_keys=True, now=NOW + 60)
    assert loaded["facts"]["occupation"]["value"] == "welder"


# ---------- §7.3 initial backfill ----------

def test_first_scope_pass_uses_backfill_window(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r
    history_key = "chat_history:platforms:webui:scope1"
    total = 300
    for i in range(total):
        r.rpush(history_key, json.dumps({"role": "user", "content": f"hello {i}", "message_id": f"m{i}", "user_id": "u1"}))

    captured = {}

    def fake_extract(llm_client, **kw):
        msgs = kw.get("messages") or []
        captured["indices"] = [
            int(str(m.get("text") or "hi -1").rsplit(" ", 1)[-1]) for m in msgs
        ]
        return {"user_observations": [], "room_observations": [], "user_removals": [], "room_removals": [],
                "user_episodes": [], "room_episodes": []}

    monkeypatch.setattr(core, "_llm_extract_observations", fake_extract)
    settings = {"lookback_limit": 50, "initial_backfill_messages": 120, "min_confidence": 0.55}
    core._process_scope(llm_client="x", platform="webui", history_key=history_key, scope_id="scope1", settings=settings)
    indices = captured["indices"]
    assert len(indices) == 50, indices
    # window starts near total - 120 (= 180); the pass covers the newest 50 of the window
    assert 180 <= indices[0] <= 230, indices[0]

    # A saved cursor is honored normally (no backfill re-kickoff)
    cursor_pos = 280
    core._save_history_cursor(
        cursor_redis_key=core.cursor_key("webui", "scope1"),
        history_key=history_key,
        position=cursor_pos,
        total_len=300,
    )
    captured.clear()
    core._process_scope(llm_client="x", platform="webui", history_key=history_key, scope_id="scope1", settings=settings)
    indices = captured["indices"]
    assert indices and indices[0] == 280


# ---------- §7.8 stock import ----------

def _seed_stock(core, r):
    stock_doc = {
        "schema_version": 2 if hasattr(core, "MEMORY_PLUS_SCHEMA_VERSION") else 1,
        "last_updated": NOW,
        "facts": {
            "occupation": {
                "value": "electrician",
                "confidence": 0.9,
                "evidence": ["s1"],
                "valid_from": NOW - 1000,
                "updated_at": NOW - 1000,
                "first_seen": NOW - 1000,
                "mentions": 4,
                "source": "stated",
                "category": "work",
                "previous": [],
            },
            "weak_fact": {
                "value": "barely there",
                "confidence": 0.3,
                "evidence": [],
                "updated_at": NOW - 1000,
                "mentions": 1,
                "source": "stated",
                "previous": [],
            },
            "favorite_color": {
                "value": "blue",
                "confidence": 0.9,
                "evidence": ["s2"],
                "updated_at": NOW - 1000,
                "mentions": 2,
                "source": "stated",
                "previous": [],
            },
        },
    }
    r.set("mem:user:webui:user1", json.dumps(stock_doc))
    r.set("mem:identity_alias:webui:user9", "aliased_person")
    r.set("mem:identity_name:Maya", "aliased_person")
    return stock_doc


def test_stock_import_is_idempotent_and_read_only(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    stock_doc = _seed_stock(core, r)
    doc = _doc_with(core, {"candidate_key": "favorite_color", "value": "blue", "confidence": 0.95})
    core.save_doc(r, core.user_doc_key("webui", "user1"), doc, now=NOW)
    stock_before = r.get("mem:user:webui:user1")

    summary = core._memory_plus_stock_import(r, min_confidence=0.55)
    assert summary["ok"] is True
    assert summary["imported_facts"] >= 1  # occupation
    assert summary["merged_facts"] == 0    # same value -> idempotent skip
    assert summary["identity_keys"] == 2

    imported = core.load_doc(r, core.user_doc_key("webui", "user1"), now=NOW)
    facts = imported["facts"]
    assert facts["occupation"]["source"] == "imported"
    assert facts["occupation"]["value"] == "electrician"
    assert facts["occupation"]["mentions"] == 4  # metadata preserved on fresh copy
    assert facts["favorite_color"]["confidence"] == 0.95  # ours untouched
    assert "weak_fact" not in facts
    assert r.get("mem:identity_alias:webui:user9") == "aliased_person"

    again = core._memory_plus_stock_import(r, min_confidence=0.55)
    assert again["imported_facts"] == 0 and again["merged_facts"] == 0
    assert r.get("mem:user:webui:user1") == stock_before


def test_stock_import_merges_divergent_value(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    stock_doc = {
        "last_updated": NOW,
        "facts": {
            "occupation": {
                "value": "nurse",
                "confidence": 0.9,
                "evidence": ["s9"],
                "updated_at": NOW - 500,
                "mentions": 3,
                "source": "stated",
                "previous": [],
            },
        },
    }
    r.set("mem:user:webui:user7", json.dumps(stock_doc))
    doc = _doc_with(core, {"candidate_key": "occupation", "value": "electrician", "confidence": 0.95})
    core.save_doc(r, core.user_doc_key("webui", "user7"), doc, now=NOW)
    before = json.loads(r.get("mem:user:webui:user7"))

    summary = core._memory_plus_stock_import(r, min_confidence=0.55)
    assert summary["merged_facts"] == 1
    merged = core.load_doc(r, core.user_doc_key("webui", "user7"), now=NOW)["facts"]["occupation"]
    # §7.4: differing value at >= existing confidence replaces and keeps previous
    assert merged["value"] == "nurse"
    assert merged["previous"] and merged["previous"][0]["value"] == "electrician"
    assert json.loads(r.get("mem:user:webui:user7")) == before
