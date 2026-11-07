"""Unit tests for the lexical retriever (spec section 6)."""

import time

import pytest

NOW = 1_800_000_000.0


def _fact(core, key, value, *, confidence=0.9, category="preferences", mentions=3, ts=NOW, source="stated"):
    return core.RetrieverDoc(id=f"fact:{key}", kind="fact", key=key, text=str(value),
                             category=category, confidence=confidence, mentions=mentions, ts=ts, source=source)


def _episode(core, text, *, category="", confidence=0.8, ts=NOW):
    return core.RetrieverDoc(id=f"episode:{text[:24]}", kind="episode", key="", text=text,
                             category=category, confidence=confidence, mentions=1, ts=ts)


@pytest.fixture
def index(core):
    docs = [
        _fact(core, "favorite_foods", "thai curry, ramen, sushi"),
        _fact(core, "occupation", "software engineer at a robotics startup", category="demographic_information"),
        _fact(core, "pet_names", "rex and clover", category="family_background"),
        _fact(core, "music_interests", "lo-fi hip hop and jazz", category="interests"),
        _fact(core, "home_thermostat", "prefers the great room at 70 degrees at night", category="home_automation"),
        _fact(core, "wife_name", "maria", category="family_background"),
    ]
    core_ret = core.get_retriever()
    return core_ret, core_ret.build_index(docs)


# ---------- tokenizer / stemmer / stopwords ----------

def test_tokenize(core):
    assert core.tokenize("Favorite_Foods: Thai Curry!") == ["favorite", "foods", "thai", "curry"]
    assert core.tokenize("") == []
    assert "2024" in core.tokenize("I moved in 2024")


def test_content_tokens_drop_stopwords(core):
    toks = core.content_tokens("Can you please tell me what I usually eat?")
    assert "can" not in toks and "you" not in toks and "please" not in toks
    assert "eat" in toks and "usually" in toks


def test_stemmer(core):
    assert core.stem("walks") == core.stem("walk") == core.stem("walking")
    assert core.stem("walking") == "walk"
    assert core.stem("studying") == "study"
    assert core.stem("classes") == "class"
    assert core.stem("cats") == "cat"
    assert core.stem("") == ""
    assert core.stem("run") == "run"


def test_stopwords_include_assistant_filler(core):
    for word in ("please", "can", "you", "tater", "hey", "thanks"):
        assert core.is_stopword(word)
    assert not core.is_stopword("curry")


# ---------- concept map ----------

def test_concept_lookup(core):
    assert "food" in core.concepts_for_token("lunch")
    assert "cooking" in core.concepts_for_token("recipe")
    assert "family" in core.concepts_for_token("daughter")
    assert "climate" in core.concepts_for_token("thermostat")
    assert core.concepts_for_token("xyzzy") == frozenset()


def test_custom_synonyms_extend_concepts(core):
    retriever = core.LexicalRetriever(core.RetrieverConfig(synonym_groups=["coffee, espresso, latte, caffeine"]))
    assert "custom_1" in retriever._concepts("latte")
    assert "custom_1" in retriever._concepts("caffeine")
    plain = core.get_retriever()
    assert "custom_1" not in plain._concepts("latte")
    assert "coffee" in plain._concepts("latte")  # builtin group still applies

    # get_retriever picks up the setting too
    settings_ret = core.get_retriever({"custom_synonyms": "bike, cycling\nbeer, lager"})
    assert settings_ret._concepts("cycling")
    settings_ret2 = core.get_retriever({"custom_synonyms": "bike, cycling\nbeer, lager"})
    assert settings_ret2 is settings_ret
    # and re-creates when the setting changes
    settings_ret3 = core.get_retriever({"custom_synonyms": "tea, oolong"})
    assert settings_ret3 is not settings_ret


# ---------- signals ----------

def test_exact_match_ranks_first(core, index, ):
    retriever, idx = index
    results = retriever.rank({"text": "what are my favorite foods", "now": NOW}, idx, k=5)
    assert results[0]["doc"].key == "favorite_foods"
    assert results[0]["score"] > 0.5


def test_morphological_match(core, index):
    retriever, idx = index
    results = retriever.rank({"text": "what jobs have I worked", "now": NOW}, idx, k=5)
    assert results[0]["doc"].key == "occupation"


def test_paraphrase_via_concepts(core, index):
    """'Cook tonight' shares no words with the stored fact; concepts must bridge."""
    retriever, idx = index
    results = retriever.rank({"text": "what should I cook tonight?", "now": NOW}, idx, k=5)
    keys = [item["doc"].key for item in results]
    assert "favorite_foods" in keys[:3]


def test_typo_trigram_fuzzy(core, index):
    retriever, idx = index
    results = retriever.rank({"text": "what do I keep in the refregerator", "now": NOW}, idx, k=5)
    # no refrigerator doc installed; ensure scoring machinery doesn't blow up
    assert isinstance(results, list)


def test_intent_boost_visible_in_breakdown(core, index):
    retriever, idx = index
    results = retriever.rank({"text": "what music do I like to listen to", "now": NOW}, idx, k=5)
    music_result = next((r for r in results if r["doc"].key == "music_interests"), None)
    assert music_result is not None
    assert music_result["breakdown"]["intent"] >= 0.0


def test_recency_decides_ties(core):
    """Near-identical docs differing only in age: the fresh one wins the tie."""
    retriever = core.get_retriever()
    fresh_docs = [
        _fact(core, "favorite_food", "ramen", ts=NOW - 400 * 86_400, confidence=0.9),
        _fact(core, "favorite_food", "sushi rolls", ts=NOW, confidence=0.9),
    ]
    idx = retriever.build_index(fresh_docs)
    results = retriever.rank({"text": "what is my favorite food", "now": NOW}, idx, k=2)
    assert results[0]["doc"].text == "sushi rolls"
    assert results[0]["breakdown"]["quality"] > results[1]["breakdown"]["quality"]


def test_short_utterance_uses_context(core, index):
    retriever, idx = index
    results = retriever.rank(
        {"text": "cook", "context_text": "what should I make for dinner tonight", "now": NOW}, idx, k=5
    )
    keys = [item["doc"].key for item in results]
    assert "favorite_foods" in keys[:3]


# ---------- selection mechanics ----------

def test_is_anchor(core):
    assert core.is_anchor(_fact(core, "wife_name", "maria"))
    assert core.is_anchor(_fact(core, "occupation", "engineer", category="demographic_information"))
    assert not core.is_anchor(_fact(core, "home_thermostat", "70 degrees"))


def test_mmr_avoids_near_duplicates(core):
    retriever = core.get_retriever()
    docs = [
        _fact(core, "favorite_foods", "spicy thai curry with thai rice and thai soup"),
        _fact(core, "favorite_dishes_dup", "spicy thai curry", category="preferences"),
        _fact(core, "tacos", "spicy tacos from the mexican place", category="preferences"),
    ]
    idx = retriever.build_index(docs)
    scored = retriever.rank({"text": "spicy thai food please", "now": NOW}, idx, k=3)
    picked = core.mmr_select(scored, k=3)
    keys = [item["doc"].key for item in picked]
    assert keys[0] in {"favorite_foods", "favorite_dishes_dup"}
    assert "tacos" in keys[:2]  # diversity pulled in a genuinely different doc


def test_breakdown_reported(core, index):
    retriever, idx = index
    results = retriever.rank({"text": "favorite foods", "now": NOW}, idx, k=1)
    assert set(results[0]["breakdown"]) == {"bm25", "trigram", "concept", "intent", "quality"}


# ---------- performance (spec 11.3) ----------

def test_benchmark_rank_and_index(core):
    import random

    random.seed(7)
    docs = []
    words = [w for words in core.CONCEPT_GROUPS.values() for w in words]
    for i in range(500):
        value = " ".join(random.choice(words) for _ in range(8))
        docs.append(_fact(core, f"dynamic_fact_{i}", value, ts=NOW - random.randint(0, 200) * 86_400))
    for i in range(200):
        text = "episode about " + " ".join(random.choice(words) for _ in range(10))
        docs.append(_episode(core, text, ts=NOW - random.randint(0, 30) * 86_400))
    retriever = core.get_retriever()
    t0 = time.perf_counter()
    idx = retriever.build_index(docs)
    build_s = time.perf_counter() - t0
    assert build_s <= 0.250, f"index build too slow: {build_s:.3f}s"

    t0 = time.perf_counter()
    for _ in range(5):
        results = retriever.rank({"text": "what should I cook tonight with my daughter", "now": NOW}, idx, k=10)
    per_call = (time.perf_counter() - t0) / 5
    assert per_call <= 0.075, f"rank too slow: {per_call * 1000:.1f}ms"
    assert results
