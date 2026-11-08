"""Phase 5: §7.7 behavior-derived habits — collectors, approval queue, injection."""

import json
import time

NOW = 1_800_000_000.0


def _ts_at(day: int, hour: int, minute: int = 30) -> float:
    """A UTC-safe local timestamp: 2026-11-(2+day) at hour:minute in this TZ."""
    return time.mktime((2026, 11, day + 2, hour, minute, 0, 0, 0, -1))


def _seed_pending(core, r, proposal_id="h_ff00", fp="ff0011", owner="", kind="awareness", subject="test · motion"):
    proposal = {
        "id": proposal_id,
        "fp": fp,
        "kind": kind,
        "subject": subject,
        "text": "(likely) Test habit pattern between 16:00–20:00 (5 events over 4 days).",
        "owner_doc_key": owner,
        "stats": {"count": 5, "days": 4, "bucket": "16:00–20:00"},
        "created_ts": NOW,
    }
    r.hset(core._HABITS_PENDING_KEY, mapping={proposal_id: json.dumps(proposal)})
    return proposal


def _seed_doc(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    doc = core.empty_memory_doc(now=NOW)
    r.hset(core.SETTINGS_KEY, mapping={})
    return r, doc


# ---------- collectors ---------

def test_awareness_collector_buckets_and_threshold(core, fake_redis_cls):
    r, _ = _seed_doc(core, fake_redis_cls)
    events = []
    for day in range(4):
        for _ in range(2):  # two events/day at ~18:30 => 8 total in one bucket
            events.append(
                json.dumps(
                    {
                        "id": "e",
                        "source": "garage",
                        "type": "camera_event",
                        "message": "Motion at garage",
                        "ha_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_ts_at(day, 18))),
                        "data": {"area": "garage", "event_type": "motion"},
                    }
                )
            )
    r.rpush("tater:automations:events:garage", *events)
    out = core._habits_collect_awareness_events(r)
    rows = [row for row in out if "garage" in row["subject"]]
    assert rows and rows[0]["kind"] == "awareness"
    bucket = core._habits_hour_bucket(_ts_at(0, 18))
    assert rows[0]["fp"] == core._habits_fingerprint("awareness", "garage · motion", bucket)
    assert rows[0]["stats"]["count"] >= 5 and rows[0]["stats"]["days"] >= 3

    # A couple of events (2) must stay under the support threshold.
    r2 = fake_redis_cls()
    r2.rpush(
        "tater:automations:events:porch",
        *events[:2],
    )
    assert core._habits_collect_awareness_events(r2) == []


def test_ga_journal_collector(core, fake_redis_cls):
    r, _ = _seed_doc(core, fake_redis_cls)
    rows = []
    for day in range(3):
        for _ in range(2):
            rows.append(
                json.dumps(
                    {
                        "ts": round(_ts_at(day, 8), 3),
                        "ts_text": "",
                        "entity": "lock.front_door",
                        "from": "unlocked",
                        "to": "locked",
                        "held_seconds": 5.0,
                    }
                )
            )
    r.rpush("automation_ga_core:journal", *rows)
    out = core._habits_collect_ga_journal(r)
    assert out and out[0]["subject"] == "front door turns locked"
    assert out[0]["stats"]["count"] >= 5 and out[0]["stats"]["days"] >= 3

    # Missing keys -> no-op
    assert core._habits_collect_ga_journal(fake_redis_cls()) == []


def test_music_collector_maps_person(core, fake_redis_cls, monkeypatch):
    r, _ = _seed_doc(core, fake_redis_cls)
    rows = []
    for day in range(3):
        for _ in range(2):
            rows.append(
                {
                    "track_id": "t",
                    "title": "Song",
                    "artist": "Norah Jones",
                    "album": "",
                    "genres": ["jazz"],
                    "person_id": "p1",
                    "person_name": "Maya",
                    "played_at": _ts_at(day, 20),
                }
            )
    r.set("music_core_listening_history_v1", json.dumps(rows))
    # No People data -> unresolvable owner -> skipped
    monkeypatch.setattr(core, "_people_person_rows", lambda redis_obj: [])
    assert core._habits_collect_music(r) == []
    monkeypatch.setattr(
        core, "_people_person_rows", lambda redis_obj: [{"id": "p1", "display_name": "Maya"}]
    )
    out = core._habits_collect_music(r)
    assert out and out[0]["owner_doc_key"] == core.identity_doc_key("p1")
    assert "Maya" in out[0]["text"] and "Norah Jones" in out[0]["text"]
    assert out[0]["kind"] == "music"
    assert out[0]["fp"] == core._habits_fingerprint(
        "music", "Maya|p1|Norah Jones", core._habits_hour_bucket(_ts_at(0, 20))
    )


def test_personal_music_collector_scoped_keys(core, fake_redis_cls, monkeypatch):
    r, _ = _seed_doc(core, fake_redis_cls)
    rows = []
    for day in range(3):
        for _ in range(2):
            rows.append(
                {"track_id": "t", "title": "Song", "artist": "Adele", "genres": ["pop"],
                 "played_at": _ts_at(day, 20), "person_name": ""}  # row omits person; key carries it
            )
    r.set("personal_music_core:history:v1:p1", json.dumps(rows))
    monkeypatch.setattr(
        core, "_people_person_rows", lambda redis_obj: [{"id": "p1", "display_name": "Maya"}]
    )
    out = core._habits_collect_personal_music(r)
    assert out and out[0]["kind"] == "personal_music"
    assert out[0]["owner_doc_key"] == core.identity_doc_key("p1")
    assert "Maya" in out[0]["text"] and "Adele" in out[0]["text"]

    # Household history (row names no person, key is the unscoped base) -> skipped
    r2 = fake_redis_cls()
    r2.set("personal_music_core:history:v1", json.dumps(rows))
    monkeypatch.setattr(
        core, "_people_person_rows", lambda redis_obj: [{"id": "p1", "display_name": "Maya"}]
    )
    assert core._habits_collect_personal_music(r2) == []


def test_both_music_cores_dedupe_to_one_proposal(core, fake_redis_cls, monkeypatch):
    r, _ = _seed_doc(core, fake_redis_cls)
    monkeypatch.setattr(
        core, "_people_person_rows", lambda redis_obj: [{"id": "p1", "display_name": "Maya"}]
    )

    def rows_for():
        return [
            {
                "track_id": "t", "title": "Song", "artist": "Coldplay", "genres": ["rock"],
                "person_id": "p1", "person_name": "Maya",
                "played_at": _ts_at(day % 3, 20),
            }
            for day in range(5)
        ]

    r.set("personal_music_core:history:v1:p1", json.dumps(rows_for()))
    r.set("music_core_listening_history_v1", json.dumps(rows_for()))
    summary = core._memory_plus_habits_run(r, {}, now_ts=NOW)
    pending = [p for p in core._memory_plus_habits_pending(r).values() if "Coldplay" in p.get("text", "")]
    assert summary["collectors"].get("personal_music", 0) >= 1
    assert summary["collectors"].get("music", 0) >= 1
    # Same person+artist+bucket in both cores -> ONE proposal (source-independent fp)
    assert len(pending) == 1


def test_chat_routines_collector(core, fake_redis_cls, monkeypatch):
    r, _ = _seed_doc(core, fake_redis_cls)
    history_key = "chat_history:platforms:webui:chat"
    msgs = []
    for day in range(3):
        for _ in range(2):
            msgs.append(
                json.dumps(
                    {
                        "role": "user",
                        "content": "what's the weather like outside?",
                        "timestamp": _ts_at(day, 7),
                        "user_id": "user9",
                        "username": "Maya",
                    }
                )
            )
    r.rpush(history_key, *msgs)
    monkeypatch.setattr(core, "_resolve_enabled_platforms", lambda: ["webui"])
    monkeypatch.setattr(core, "_discover_scopes", lambda platform: [(history_key, "chat")])
    out = core._habits_collect_chat_routines(r)
    assert out
    row = out[0]
    assert row["kind"] == "chat_routine"
    assert row["owner_doc_key"].startswith("advanced_memory_core:user:")
    assert "the weather" in row["text"]
    assert row["stats"]["count"] >= 5 and row["stats"]["days"] >= 3


# ---------- nightly run ----------

def test_habits_run_dedupes_dismisses_and_skips_approved(core, fake_redis_cls, monkeypatch):
    r, doc = _seed_doc(core, fake_redis_cls)
    doc_key = core.user_doc_key("webui", "user1")
    core.save_doc(r, doc_key, doc, now=NOW)

    def fake_collect(_r):
        return [
            {
                "fp": "aaaa000000000001",
                "kind": "awareness",
                "subject": "garage · motion",
                "text": "(likely) pattern A.",
                "owner_doc_key": "",
                "stats": {"count": 6, "days": 5, "bucket": "16:00–20:00"},
            },
            {   # approved already in the owner doc
                "fp": "aaaa000000000002",
                "kind": "chat_routine",
                "subject": "asks about the weather",
                "text": "(likely) pattern B.",
                "owner_doc_key": doc_key,
                "stats": {"count": 6, "days": 5, "bucket": "04:00–08:00"},
            },
            {   # dismissed fingerprint (recorded below)
                "fp": "aaaa000000000003",
                "kind": "ga_journal",
                "subject": "door turns closed",
                "text": "(likely) pattern C.",
                "owner_doc_key": "",
                "stats": {"count": 6, "days": 5, "bucket": "00:00–04:00"},
            },
        ]

    approved_doc = core.load_doc(r, doc_key, now=NOW)
    approved_doc["facts"]["habit_x"] = {"value": "v", "habit_fp": "aaaa000000000002"}
    core.save_doc(r, doc_key, approved_doc, now=NOW)
    r.hset(core._HABITS_DISMISSED_KEY, mapping={"aaaa000000000003": "1"})

    monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: fake_collect(r))
    summary = core._memory_plus_habits_run(r, {}, now_ts=NOW)
    assert summary["proposed"] >= 1
    pending = core._memory_plus_habits_pending(r)
    fps = {p.get("fp") for p in pending.values()}
    assert "aaaa000000000001" in fps          # new -> proposed
    assert "aaaa000000000002" not in fps      # already approved -> skipped
    assert "aaaa000000000003" not in fps      # dismissed -> skipped

    # Second pass proposes nothing new (dedupe by pending fingerprints).
    again = core._memory_plus_habits_run(r, {}, now_ts=NOW + 10)
    assert again["proposed"] == 0
    assert again["pending"] == len(pending)


# ---------- approval queue actions ----------

def _run(core, r, values):
    return core.handle_htmlui_tab_action(
        action="memory_habits", payload={"values": values}, redis_client=r, core_key="advanced_memory_core"
    )


def test_approve_flow_merges_inferred_fact(core, fake_redis_cls):
    r, _ = _seed_doc(core, fake_redis_cls)
    doc_key = core.user_doc_key("webui", "user1")
    proposal = _seed_pending(core, r, owner="")
    core.save_doc(r, doc_key, core.empty_memory_doc(now=NOW), now=NOW)

    # Approve without an owner -> rejected (home-wide patterns need an owner choice)
    try:
        _run(core, r, {"habit_id": "h_ff00", "habit_decision": "approve", "habit_owner_doc": ""})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    out = _run(core, r, {"habit_id": "h_ff00", "habit_decision": "approve", "habit_owner_doc": doc_key})
    assert out["ok"] is True
    fact = core.load_doc(r, doc_key, now=NOW)["facts"][f"habit_{proposal['fp']}"]
    assert fact["source"] == "inferred" and fact["habit_fp"] == proposal["fp"]
    assert fact["category"] == "behavioral_patterns" and fact["confidence"] == 0.65
    assert core._memory_plus_habits_pending(r) == {}

    # "(likely)" marking flows end-to-end into a matching query's injection fragment
    payload = core.get_hydra_memory_context_payload(
        platform="webui",
        scope="chat",
        origin={"platform": "webui", "user_id": "user1"},
        redis_client=r,
        user_text="test habit pattern around evening",
        request_text="",
    )
    frags = core.get_hydra_system_prompt_fragments(
        role="chat",
        platform="webui",
        scope="chat",
        origin={"platform": "webui", "user_id": "user1"},
        redis_client=r,
        memory_context=payload,
        user_text="test habit pattern around evening",
        request_text="",
    )
    chat = "\n".join(frags.get("chat") or [])
    items = (payload.get("user") or {}).get("items") or []
    assert any("(likely)" in item.get("injected_text", "") for item in items) or "(likely)" in chat


def test_dismiss_flow_blocks_reproposal(core, fake_redis_cls, monkeypatch):
    r, _ = _seed_doc(core, fake_redis_cls)
    _seed_pending(core, r, proposal_id="h_ff00", fp="ff0011")
    out = _run(core, r, {"habit_id": "h_ff00", "habit_decision": "dismiss"})
    assert out["ok"] is True
    assert core._memory_plus_habits_pending(r) == {}
    assert "ff0011" in core._memory_plus_habits_dismissed_fps(r)

    pattern = {
        "fp": "ff0011",
        "kind": "awareness",
        "subject": "test · motion",
        "text": "t",
        "owner_doc_key": "",
        "stats": {"count": 6, "days": 5, "bucket": "00:00–04:00"},
    }
    monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: [pattern])
    summary = core._memory_plus_habits_run(r, {}, now_ts=NOW)
    assert summary["proposed"] == 0


def test_keep_keep_pending_and_bad_inputs(core, fake_redis_cls):
    r, _ = _seed_doc(core, fake_redis_cls)
    _seed_pending(core, r)
    out = _run(core, r, {"habit_id": "h_ff00", "habit_decision": "keep"})
    assert out["ok"] is True
    assert "h_ff00" in core._memory_plus_habits_pending(r)

    for values in (
        {},
        {"habit_id": "h_nope"},  # KeyError: unknown proposal
        {"habit_id": "h_ff00", "habit_decision": "bogus"},
        {"habit_id": "h_ff00", "habit_decision": "approve", "habit_owner_doc": "mem:user:x"},
    ):
        try:
            _run(core, r, values)
            raise AssertionError(f"expected error for {values}")
        except (ValueError, KeyError):
            pass


# ---------- run-loop wiring ----------

def test_cycle_wiring_runs_habits_when_enabled_and_due(core, fake_redis_cls, monkeypatch):
    r, _ = _seed_doc(core, fake_redis_cls)
    pattern = {
        "fp": "cccc000000000001",
        "kind": "awareness",
        "subject": "garage · motion",
        "text": "t",
        "owner_doc_key": "",
        "stats": {"count": 6, "days": 5, "bucket": "00:00–04:00"},
    }
    monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: [pattern])
    monkeypatch.setattr(core, "_habits_collect_ga_journal", lambda redis_obj: [])
    monkeypatch.setattr(core, "_habits_collect_music", lambda redis_obj: [])
    monkeypatch.setattr(core, "_habits_collect_chat_routines", lambda redis_obj: [])

    settings = {
        "inject_context": "false",            # shadow: keep payload empty during the test
        "enable_consolidation": "false",
        "enable_inferred_habits": "true",
        "consolidation_hour": time.localtime().tm_hour,  # match the current hour -> due now
        "min_confidence": 0.55,
        "lookback_limit": 10,
        "initial_backfill_messages": 0,
    }
    out = core._run_memory_cycle_once(None, settings)
    assert out["ok"] is True
    summary = json.loads(r.hget("advanced_memory_core:stats", "last_habits_summary") or "{}")
    assert summary.get("proposed") == 1
    assert r.hget("advanced_memory_core:stats", "last_habits_ts")

    # A second cycle right after must not re-run (daily throttle).
    ts_before = r.hget("advanced_memory_core:stats", "last_habits_ts")
    core._run_memory_cycle_once(None, settings)
    assert r.hget("advanced_memory_core:stats", "last_habits_ts") == ts_before

    # And with the setting OFF, nothing runs (timestamp stored but stays "0.0").
    r2 = fake_redis_cls()
    core.redis_client = r2
    core._run_memory_cycle_once(None, {**settings, "enable_inferred_habits": "false"})
    assert (r2.hget("advanced_memory_core:stats", "last_habits_ts") or "0.0") == "0.0"
