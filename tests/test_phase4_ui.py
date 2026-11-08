"""Phase 4: WebUI tab data + tab actions (spec 10)."""

import json

NOW = 1_800_000_000.0


def _seed(core, fake_redis_cls, episodes=True):
    r = fake_redis_cls()
    core.redis_client = r
    doc = core.empty_memory_doc(now=NOW)
    core.merge_observation(doc, {"candidate_key": "favorite_foods", "value": "thai curry, ramen, sushi",
                                 "confidence": 0.9, "category": "preferences"}, min_confidence=0.5,
                           default_ttl_sec=0, allow_new_keys=True, now=NOW)
    core.merge_observation(doc, {"candidate_key": "wife_name", "value": "Priya",
                                 "confidence": 0.95, "category": "family_background"}, min_confidence=0.5,
                           default_ttl_sec=0, allow_new_keys=True, now=NOW)
    doc_key = core.user_doc_key("webui", "user1")
    core.save_doc(r, doc_key, doc, now=NOW)
    if episodes:
        core._memory_store_episodes(
            r, doc_key,
            [{"id": "ep1", "summary": "Asked to automate the garage door at night",
              "topics": ["garage"], "salience": 0.8, "status": "historical", "occurred_at": NOW - 3 * 86_400}],
            min_salience=0.2, max_items=50, now_ts=NOW,
        )
    return r, doc_key


# ---------- tab data shape ----------

def test_tab_data_manager_shape(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    data = core.get_htmlui_tab_data(redis_client=r)
    ui = data["ui"]
    assert ui["kind"] == "settings_manager"
    tab_keys = {tab["key"] for tab in ui["manager_tabs"]}
    assert {"overview", "people", "rooms", "tools"} <= tab_keys
    assert ui["add_form"]["action"] == "memory_tool_action"
    groups = {f.get("group") for f in ui["item_forms"]}
    assert "overview" in groups and "person" in groups
    stats = {item["label"]: item["value"] for item in data["stats"]}
    assert stats["Facts"] >= 2


def test_tab_data_marks_shadow_mode(core, fake_redis_cls):
    r, _ = _seed(core, fake_redis_cls)
    r.hset("advanced_memory_core_settings", mapping={"inject_context": "false"})
    data = core.get_htmlui_tab_data(redis_client=r)
    assert any(item.get("value") == "shadow mode (extraction only, no injection)" for item in data["stats"])


def test_tab_conflict_warning_on_stock_running(core, fake_redis_cls):
    r, _ = _seed(core, fake_redis_cls)
    r.set("memory_core_running", "1")
    status = core._memory_plus_conflict_status(r, core._settings(r))
    assert status["stock_running"] is True and status["warning"].startswith("CONFLICT")
    data = core.get_htmlui_tab_data(redis_client=r)
    overview = next(f for f in data["ui"]["item_forms"] if f.get("group") == "overview")
    assert "CONFLICT" in overview["subtitle"]


def test_doc_rows_include_facts_episodes_and_card(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    rows = core._memory_plus_doc_rows(r)
    row = next(row for row in rows if row["doc_key"] == doc_key)
    assert row["fact_count"] >= 2
    assert row["episodes"] and row["episodes"][0]["id"] == "ep1"
    assert row["profile_card"]


# ---------- recall tester action ----------

def _run_action(core, r, action, values):
    return core.handle_htmlui_tab_action(action=action, payload={"values": values}, redis_client=r, core_key="advanced_memory_core")


def test_recall_tester_reports_ranking_and_injection(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    out = _run_action(core, r, "memory_tool_action", {
        "tool_action": "recall_test",
        "tester_query": "what should I cook tonight",
        "tester_doc_key": doc_key,
    })
    assert out["ok"] is True
    text = out["message"]
    assert "Ranked results" in text and "Would be injected" in text
    assert "breakdown" not in text  # breakdown keys are shown as bm25=/tri= pairs
    assert "bm25=" in text and "concept=" in text
    assert "Fragment length:" in text


def test_recall_tester_requires_query_and_doc(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    for values in ({"tool_action": "recall_test"}, {"tool_action": "recall_test", "tester_query": "hi"}):
        try:
            _run_action(core, r, "memory_tool_action", values)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_recall_tester_shadow_mode_says_nothing_injected(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    r.hset("advanced_memory_core_settings", mapping={"inject_context": "false"})
    out = _run_action(core, r, "memory_tool_action", {
        "tool_action": "recall_test", "tester_query": "what should I cook tonight",
        "tester_doc_key": doc_key,
    })
    assert out["ok"] is True and "would inject nothing" in out["message"]


# ---------- manage doc action ----------

def test_manage_doc_edit_and_forget(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls, episodes=True)
    out = _run_action(core, r, "memory_manage_doc", {
        "doc_key": doc_key, "edit_key": "favorite_foods", "edit_value": "ramen only",
    })
    assert out["ok"] is True
    doc = core.load_doc(r, doc_key, now=NOW)
    fact = doc["facts"]["favorite_foods"]
    assert fact["value"] == "ramen only" and fact["source"] == "manual"
    assert fact["previous"] and fact["previous"][0]["value"] == "thai curry, ramen, sushi"

    out2 = _run_action(core, r, "memory_manage_doc", {"doc_key": doc_key, "forget_keys": "occupation, favorite_foods"})
    assert out2["ok"] is True and "1 fact" in out2["message"]
    doc2 = core.load_doc(r, doc_key, now=NOW)
    assert "occupation" not in doc2["facts"] and "favorite_foods" not in doc2["facts"]


def test_manage_doc_rejects_bad_ids(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    for values in (
        {"doc_key": "mem:user:nope", "forget_keys": "x"},
        {"doc_key": doc_key, "forget_keys": "nope_key_that_is_not_there"},
        {"doc_key": doc_key, "forget_keys": "favorite_foods", "forget_episode_id": "missing_id"},
    ):
        try:
            _run_action(core, r, "memory_manage_doc", values)
            raise AssertionError(f"expected error for {values}")
        except (ValueError, KeyError):
            pass


def test_manage_doc_forgets_episode(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    out = _run_action(core, r, "memory_manage_doc", {"doc_key": doc_key, "forget_episode_id": "ep1"})
    assert out["ok"] is True
    assert core._memory_load_episodes(r, doc_key) == []


# ---------- maintenance actions ----------

def test_maintenance_import_and_backup_roundtrip(core, fake_redis_cls, tmp_path, monkeypatch):
    r, doc_key = _seed(core, fake_redis_cls)
    monkeypatch.setattr(core, "_MEMORY_UI_EXPORT_DIR", str(tmp_path / "exports"))

    out = _run_action(core, r, "memory_tool_action", {"tool_action": "export_backup"})
    assert out["ok"] is True
    files = list((tmp_path / "exports").glob("*.json"))
    assert files, "backup file was written"

    r2 = fake_redis_cls()
    backup_text = files[0].read_text()
    assert backup_text.startswith("{")  # valid JSON exported
    out2 = _run_action(core, r2, "memory_tool_action", {
        "tool_action": "restore_backup", "backup_json": backup_text,
    })
    assert out2["ok"] is True and "Restored" in out2["message"]
    r2_doc = core.load_doc(r2, doc_key, now=NOW)
    assert r2_doc["facts"]["favorite_foods"]["value"] == "thai curry, ramen, sushi"


def test_maintenance_wipe_requires_confirmation(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    try:
        _run_action(core, r, "memory_tool_action", {"tool_action": "wipe_all", "wipe_confirm": "wipe please"})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    out = _run_action(core, r, "memory_tool_action", {"tool_action": "wipe_all", "wipe_confirm": "WIPE"})
    assert out["ok"] is True
    assert not r.keys("advanced_memory_core:*") or r.keys("advanced_memory_core:*") == []
    assert r.exists("advanced_memory_core_settings") == 0
    assert r.keys("mem:user:*") or not r.keys("mem:user:*")


def test_unsupported_actions_raise_key_error(core, fake_redis_cls):
    r, doc_key = _seed(core, fake_redis_cls)
    for action, values in (
        ("bogus_action", {}),
        ("memory_tool_action", {"tool_action": "not_a_tool"}),
        ("memory_maintenance", {}),
    ):
        try:
            _run_action(core, r, action, values)
            raise AssertionError(f"expected KeyError for {action}")
        except KeyError:
            pass
