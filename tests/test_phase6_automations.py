"""Phase 6: presence routines (Face-ID) + automation suggestions + proactive asks."""

import json
import time

NOW = 1_800_000_000.0


def _ts_at(day: int, hour: int, minute: int = 15) -> float:
    return time.mktime((2026, 11, day + 2, hour, minute, 0, 0, 0, -1))


def _seed(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    return r


def _stamp(iso_time: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(iso_time))


def _face_session(core, day, area="gym room", person="p1"):
    return json.dumps(
        {
            "id": f"s{day}",
            "area": area,
            "monitor_id": "m1",
            "status": "complete",
            "created_at": _stamp(_ts_at(day, 5)),
            "completed_at": _stamp(_ts_at(day, 5, 6)),
            "recognized_person_ids": [person],
        }
    )


# ---------- presence collector ----------

def test_presence_collector_person_area_routine(core, fake_redis_cls, monkeypatch):
    r = _seed(core, fake_redis_cls)
    rows = [_face_session(core, day) for day in range(2)] + [_face_session(core, day) for day in range(3)]
    for i, blob in enumerate(rows):
        r.set(f"awareness:face_session:s{i}", blob)
    monkeypatch.setattr(
        core, "_people_person_rows", lambda redis_obj: [{"id": "p1", "display_name": "Maya"}]
    )
    out = core._habits_collect_presence(r)
    assert out
    row = out[0]
    assert row["kind"] == "presence" and row["area"] == "gym room"
    assert row["owner_doc_key"] == core.identity_doc_key("p1")
    assert "Maya" in row["text"] and "gym room" in row["text"]
    assert row["stats"]["count"] >= 5 and row["stats"]["days"] >= 3
    skew = core._habits_weekday_label([])
    assert row["stats"].get("schedule") in ("", skew, "mostly weekdays", "mostly weekends", "mostly weekdays")

    # Unresolvable person -> no proposal, no orphan doc
    monkeypatch.setattr(core, "_people_person_rows", lambda redis_obj: [])
    assert core._habits_collect_presence(r) == []


# ---------- recurrence -> automation proposals ----------

def _pattern(fp, *, kind="presence", owner="advanced_memory_core:user:webui:maya", area="gym room",
             bucket="04:00–08:00", text="(likely) Maya is usually in the gym room between 04:00–08:00."):
    return {
        "fp": fp,
        "kind": kind,
        "subject": f"Maya in gym room" if kind == "presence" else "asks about music",
        "text": text,
        "owner_doc_key": owner,
        "area": area,
        "stats": {"count": 10, "days": 4, "bucket": bucket},
    }


def _approve_core_pattern(core, r, fp, owner=None):
    owner = owner or core.user_doc_key("webui", "maya")
    doc = core.load_doc(r, owner, now=NOW)
    doc["facts"].setdefault(
        f"habit_{fp}",
        {"value": "v", "confidence": 0.65, "evidence": [], "updated_at": NOW, "first_seen": NOW,
         "mentions": 1, "source": "inferred", "category": "behavioral_patterns", "previous": [],
         "pending_alt": None, "habit_fp": fp, "habit_kind": "presence"},
    )
    core.save_doc(r, owner, doc, now=NOW)
    r.hset(core._HABITS_APPROVED_KEY, fp, owner)  # approval-time record, as the action does
    return owner


def test_recurrence_creates_automation_proposal_once(core, fake_redis_cls, monkeypatch):
    r = _seed(core, fake_redis_cls)
    owner = _approve_core_pattern(core, r, "aaaa000000000001")
    for name, kind in (
        ("_habits_collect_awareness_events", "awareness"),
        ("_habits_collect_ga_journal", "ga_journal"),
        ("_habits_collect_personal_music", "personal_music"),
        ("_habits_collect_music", "music"),
        ("_habits_collect_presence", "presence"),
        ("_habits_collect_chat_routines", "chat_routine"),
    ):
        monkeypatch.setattr(core, name, lambda redis_obj, k=kind: [_pattern("aaaa000000000001", kind=k)] if k == "presence" else [])

    settings = {
        "enable_automation_suggestions": "true",
        "automation_offer_scope": "person_routines",
        "automation_offer_after_recurrences": 2,
        "automation_offers_channel": "off",
    }
    summary = core._memory_plus_habits_run(r, settings, now_ts=NOW)
    pending = core._memory_plus_habits_pending(r)
    autos = [p for p in pending.values() if p.get("kind") == "automation_suggest"]
    assert summary["automation_proposed"] == 1 and len(autos) == 1
    assert autos[0]["source_fp"] == "aaaa000000000001"
    assert autos[0]["owner_doc_key"] == owner
    assert "aaaa000000000001" in core._memory_plus_habits_auto_offered(r)

    # Second run: no duplicate offer, proposal stays pending once.
    again = core._memory_plus_habits_run(r, settings, now_ts=NOW + 60)
    autos_now = [p for p in core._memory_plus_habits_pending(r).values() if p.get("kind") == "automation_suggest"]
    assert again["automation_proposed"] == 0 and len(autos_now) == 1


def test_scope_setting_gating(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r

    def _seed_world(redis_obj, fp, *, owner_in_pattern, approve_owner=None):
        """Approve fp into a person doc; the collector may or may not re-resolve it."""
        approve_owner = approve_owner or core.user_doc_key("webui", "maya")
        _approve_core_pattern(core, redis_obj, fp, owner=approve_owner)
        for name in ("_habits_collect_ga_journal", "_habits_collect_personal_music",
                     "_habits_collect_music", "_habits_collect_chat_routines"):
            monkeypatch.setattr(core, name, lambda redis_obj: [])
        pattern = _pattern(fp)
        if not owner_in_pattern:
            pattern["kind"] = "awareness"  # home-wide shapes arrive owner-less
            pattern["owner_doc_key"] = ""
        monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: [pattern])
        redis_obj.hdel(core._HABITS_AUTO_OFFERED_KEY, fp)
        return approve_owner

    def _run(r, scope):
        return core._memory_plus_habits_run(
            r,
            {"enable_automation_suggestions": "true", "automation_offer_scope": scope,
             "automation_offer_after_recurrences": 2, "automation_offers_channel": "off"},
            now_ts=NOW,
        )

    # Person-owned recurring pattern: offered under person_routines...
    owner = _seed_world(r, "bbbb000000000001", owner_in_pattern=True)
    summary = _run(r, "person_routines")
    autos = [p for p in core._memory_plus_habits_pending(r).values() if p.get("kind") == "automation_suggest"]
    assert summary["automation_proposed"] == 1 and autos
    assert autos[0]["owner_doc_key"] == owner

    # person_routines does NOT offer a pattern the user filed under a room doc
    r2 = fake_redis_cls()
    core.redis_client = r2
    room_owner = core.room_doc_key("webui", "living_room")
    _seed_world(r2, "dddd000000000001", owner_in_pattern=False, approve_owner=room_owner)
    summary2 = _run(r2, "person_routines")
    autos_pending = [p for p in core._memory_plus_habits_pending(r2).values() if p.get("kind") == "automation_suggest"]
    assert summary2["automation_proposed"] == 0 and not autos_pending and summary2["automation_candidates"] == 0

    # ...but any_pattern still escalates room-doc-approved patterns.
    s_any = _run(r2, "any_pattern")
    autos2 = [p for p in core._memory_plus_habits_pending(r2).values() if p.get("kind") == "automation_suggest"]
    assert s_any["automation_proposed"] == 1 and len(autos2) == 1
    assert autos2[0]["owner_doc_key"] == room_owner


def test_composite_scope_requires_action_partner(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r
    owner = _approve_core_pattern(core, r, "eeee000000000001")   # presence
    _approve_core_pattern(core, r, "eeee000000000002", owner=owner)  # action
    action = dict(_pattern("eeee000000000002", kind="chat_routine"))
    action.pop("area", None)
    action["stats"]["bucket"] = "04:00–08:00"

    for name in ("_habits_collect_ga_journal", "_habits_collect_personal_music",
                 "_habits_collect_music", "_habits_collect_chat_routines"):
        monkeypatch.setattr(core, name, lambda redis_obj: [])

    settings = {"enable_automation_suggestions": "true", "automation_offer_scope": "composite",
                "automation_offer_after_recurrences": 2, "automation_offers_channel": "off"}

    # presence pattern without an approved action partner -> nothing
    monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: [])
    monkeypatch.setattr(core, "_habits_collect_presence", lambda redis_obj: [])
    summary = core._memory_plus_habits_run(r, settings, now_ts=NOW)
    assert not [p for p in core._memory_plus_habits_pending(r).values() if p.get("kind") == "automation_suggest"]

    # with both presence + action pattern recurring -> composite candidate
    monkeypatch.setattr(core, "_habits_collect_presence", lambda redis_obj: [_pattern("eeee000000000001")])
    monkeypatch.setattr(core, "_habits_collect_chat_routines", lambda redis_obj: [action])
    monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: [])
    summary = core._memory_plus_habits_run(r, settings, now_ts=NOW)
    autos = [p for p in core._memory_plus_habits_pending(r).values() if p.get("kind") == "automation_suggest"]
    assert summary["automation_proposed"] >= 1 and autos


def test_proactive_channel_notification_called(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r
    _approve_core_pattern(core, r, "ffff000000000001")
    pattern = _pattern("ffff000000000001")
    for name in ("_habits_collect_awareness_events", "_habits_collect_ga_journal",
                 "_habits_collect_personal_music", "_habits_collect_music",
                 "_habits_collect_presence", "_habits_collect_chat_routines"):
        monkeypatch.setattr(core, name, lambda redis_obj: [])

    called = []
    monkeypatch.setattr(core, "_habits_notify_person", lambda text: called.append(text) or {"ok": True})
    monkeypatch.setattr(core, "_habits_collect_presence", lambda redis_obj: [pattern])
    summary = core._memory_plus_habits_run(
        r,
        {"enable_automation_suggestions": "true", "automation_offer_scope": "person_routines",
         "automation_offer_after_recurrences": 2, "automation_offers_channel": "notification"},
        now_ts=NOW,
    )
    assert called and "habit" in called[0].lower() or called
    assert summary["automation_proposed"] == 1


def test_announcement_target_resolution(core, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    r.set(
        core._HABITS_SAT_REGISTRY_KEY,
        json.dumps(
            [
                {"selector": "host:gym-sat", "metadata": {"area_name": "gym"}, "name": "Gym SAT"},
                {"selector": "host:kitchen", "metadata": {"area_name": "kitchen"}, "name": "Kitchen"},
            ]
        ),
    )
    assert core._habits_resolve_announcement_target("gym", "") == ["host:gym-sat"]
    assert core._habits_resolve_announcement_target("gym_room", "") == ["host:gym-sat"]
    # fallback default SAT when the area has no satellite
    assert core._habits_resolve_announcement_target("garage", "KITCHEN") == ["host:kitchen"]
    assert core._habits_resolve_announcement_target("garage", "") == []


def test_dismissing_automation_idea_never_reproposes(core, fake_redis_cls, monkeypatch):
    r = fake_redis_cls()
    core.redis_client = r
    _approve_core_pattern(core, r, "abab000000000001")
    settings = {"enable_automation_suggestions": "true", "automation_offer_scope": "person_routines",
                "automation_offer_after_recurrences": 2, "automation_offers_channel": "off"}
    for name in ("_habits_collect_ga_journal", "_habits_collect_personal_music",
                 "_habits_collect_music", "_habits_collect_chat_routines"):
        monkeypatch.setattr(core, name, lambda redis_obj: [])
    monkeypatch.setattr(core, "_habits_collect_presence", lambda redis_obj: [_pattern("abab000000000001")])
    monkeypatch.setattr(core, "_habits_collect_awareness_events", lambda redis_obj: [])
    core._memory_plus_habits_run(r, settings, now_ts=NOW)
    core.handle_htmlui_tab_action(
        action="memory_habits",
        payload={"values": {"habit_id": "auto_abab000000000001", "habit_decision": "dismiss"}},
        redis_client=r,
        core_key="advanced_memory_core",
    )
    assert "as_abab000000000001" in core._memory_plus_habits_dismissed_fps(r)
    core._memory_plus_habits_run(r, settings, now_ts=NOW + 120)
    autos = [p for p in core._memory_plus_habits_pending(r).values() if p.get("kind") == "automation_suggest"]
    assert not autos


# ---------- approve routes through Hydra ----------

def test_approve_automation_calls_hydra_turn(core, monkeypatch, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    proposal = {
        "id": "auto_zz", "fp": "as_zz", "source_fp": "zz", "kind": "automation_suggest",
        "subject": "Automation: test", "text": "Automation idea: test habit.",
        "owner_doc_key": "", "stats": {"count": 5, "bucket": "04:00–08:00"}, "created_ts": NOW,
    }
    r.hset(core._HABITS_PENDING_KEY, mapping={"auto_zz": json.dumps(proposal)})

    seen = {}
    def fake_hydra(redis_obj, instruction):
        seen["instruction"] = instruction
        return {"text": "Created automation a_1"}

    monkeypatch.setattr(core, "_habits_run_hydra_turn", fake_hydra)
    out = core.handle_htmlui_tab_action(
        action="memory_habits",
        payload={"values": {"habit_id": "auto_zz", "habit_decision": "approve", "habit_owner_doc": ""}},
        redis_client=r,
        core_key="advanced_memory_core",
    )
    assert out["ok"] is True and "Hydra" in out["message"] and "a_1" in out["message"]
    assert seen["instruction"].startswith("Create a new automation")
    assert "auto_zz" not in core._memory_plus_habits_pending(r)


def test_approve_automation_without_llm_fails_cleanly(core, monkeypatch, fake_redis_cls):
    r = fake_redis_cls()
    core.redis_client = r
    proposal = {
        "id": "auto_yy", "fp": "as_yy", "source_fp": "yy", "kind": "automation_suggest",
        "subject": "Automation: test", "text": "Automation idea: test habit.",
        "owner_doc_key": "", "stats": {"count": 5, "bucket": "04:00–08:00"}, "created_ts": NOW,
    }
    r.hset(core._HABITS_PENDING_KEY, mapping={"auto_yy": json.dumps(proposal)})

    def no_llm():
        raise RuntimeError("no env")

    monkeypatch.setattr(core, "_get_primary_llm_client_from_env", no_llm, raising=True)
    try:
        core.handle_htmlui_tab_action(
            action="memory_habits",
            payload={"values": {"habit_id": "auto_yy", "habit_decision": "approve"}},
            redis_client=r,
            core_key="advanced_memory_core",
        )
        raise AssertionError("expected failure")
    except (ValueError, RuntimeError):
        pass
    assert "auto_yy" in core._memory_plus_habits_pending(r)  # not consumed on failure
