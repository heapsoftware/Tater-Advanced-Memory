"""Test fixtures for memory_plus_core: stub Tater runtime modules, then load the core."""

import importlib.util
import json
import sys
import threading
import time
import types

import pytest

CORE_PATH = "/home/tater-advanced-memory/tater-memory-plus/cores/memory_plus_core.py"


class FakeRedis:
    """Minimal in-memory Redis double covering the core's surface."""

    def __init__(self):
        self._strings = {}
        self._hashes = {}
        self._lists = {}
        self.lock = threading.Lock()

    # ---- strings
    def get(self, key):
        return self._strings.get(key)

    def set(self, key, value, *args, **kwargs):
        self._strings[key] = str(value)
        return True

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            if self._strings.pop(key, None) is not None:
                deleted += 1
            if self._hashes.pop(key, None) is not None:
                deleted += 1
            if self._lists.pop(key, None) is not None:
                deleted += 1
        return deleted

    def exists(self, *keys):
        total = 0
        for key in keys:
            if key in self._strings or key in self._hashes or key in self._lists:
                total += 1
        return total

    def keys(self, pattern="*"):
        import fnmatch

        all_keys = set(self._strings) | set(self._hashes) | set(self._lists)
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    def scan_iter(self, match="*", count=None, **kwargs):
        for key in self.keys(match):
            yield key

    # ---- hashes
    def hgetall(self, key):
        return dict(self._hashes.get(key, {}))

    def hset(self, key, field=None, value=None, mapping=None):
        h = self._hashes.setdefault(key, {})
        if mapping:
            k, v = str(field or ""), value
            for fk, fv in mapping.items():
                h[str(fk)] = str(fv)
        if field is not None:
            h[str(field)] = str(value)
        return 1

    def hget(self, key, field):
        return self._hashes.get(key, {}).get(str(field))

    def hdel(self, key, *fields):
        h = self._hashes.get(key, {})
        removed = 0
        for field in fields:
            if h.pop(str(field), None) is not None:
                removed += 1
        return removed

    def hexists(self, key, field):
        return str(field) in self._hashes.get(key, {})

    # ---- lists
    def rpush(self, key, *values):
        self._lists.setdefault(key, []).extend(values)
        return len(self._lists[key])

    def lrange(self, key, start, stop):
        # Redis LRANGE is INCLUSIVE of stop; Python slices are not.
        items = self._lists.get(key, [])
        if stop < 0:
            stop = len(items) + stop
        return list(items[start : stop + 1])

    def llen(self, key):
        return len(self._lists.get(key, []))


@pytest.fixture
def fake_redis_cls():
    return FakeRedis


def _install_helpers_stub(monkeypatch_extra=None):
    helpers = types.ModuleType("helpers")
    helpers.redis_client = FakeRedis()
    helpers.extract_json = staticmethod(lambda *a, **k: None)
    helpers.get_llm_client_from_env = staticmethod(lambda *a, **k: None)

    def fake_structured_chat(*args, **kwargs):
        return {"observations": [], "user_removals": [], "episodes": []}

    helpers.llm_structured_chat = staticmethod(fake_structured_chat)
    helpers.get_primary_llm_client_from_env = staticmethod(
        lambda *a, **k: helpers.get_llm_client_from_env()
    )
    sys.modules["helpers"] = helpers

    try:
        import people  # noqa: F401

        has_people = True
    except Exception:
        has_people = False
    if not has_people:
        # Leave "people" importable=False so _people_api_module degrades gracefully.
        sys.modules.setdefault("people", None)
    return helpers


def load_core(unique_name="memory_plus_core_test"):
    _install_helpers_stub()
    spec = importlib.util.spec_from_file_location(unique_name, CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def core():
    return load_core()


@pytest.fixture
def core_fresh():
    """Load the core under a new unique module name (state isolation)."""
    name = f"memory_plus_core_test_{time.time_ns()}"
    return load_core(name)


class ScriptedLLM:
    """Fake llm client returning canned structured-chat payloads per call."""

    def __init__(self, payloads=None):
        self.payloads = list(payloads or [])
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.payloads:
            item = self.payloads.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, (list, tuple)) else (item,)
        return ({},)
