"""Retrieval evaluation harness (spec 11.3): new scorer vs stock-scorer baseline.

Fixture: a fictional person with 62 facts + 40 episodes and 84 labeled queries
across exact / morphological / paraphrase / typo / short_utterance / episode /
negative groups. Reports Recall@5, Recall@10, MRR per group plus a markdown
table (printed to stdout for the README).
"""

import json
import math
import time
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "eval_fixture.json"
NOW = 1_800_000_000.0


def _load_fixture():
    return json.loads(FIXTURE_PATH.read_text())


def _fact_docs(core, fixture):
    docs = []
    for row in fixture["facts"]:
        age_days = {"demographic_information": 30, "behavioral_patterns": 20}.get(row["category"], 45)
        docs.append(
            core.RetrieverDoc(
                id=row["key"],
                kind="fact",
                key=row["key"],
                text=str(row["value"]),
                category=row.get("category", ""),
                confidence=float(row.get("confidence", 0.9)),
                mentions=int(row.get("mentions", 3)),
                ts=NOW - age_days * 86_400,
                source="stated",
            )
        )
    return docs


def _episode_docs(core, fixture):
    docs = []
    for i, row in enumerate(fixture["episodes"]):
        docs.append(
            core.RetrieverDoc(
                id=f"episode:{i}",
                kind="episode",
                key="",
                text=row["text"],
                category=",".join(row.get("topics", [])),
                confidence=0.8,
                mentions=1,
                ts=NOW - float(row["ts_days_ago"]) * 86_400,
                source="inferred",
            )
        )
    return docs


def _all_docs(core, fixture):
    return _fact_docs(core, fixture) + _episode_docs(core, fixture)


def _relevant_for_query(query):
    if "relevant_episode_texts" in query:
        return [("episode:", text) for text in query["relevant_episode_texts"]]
    return [("", key) for key in query.get("relevant", [])]


def _is_hit(doc, relevant):
    for prefix, target in relevant:
        if prefix == "episode:":
            if doc.kind == "episode" and doc.raw_text == target:
                return True
        else:
            if doc.kind == "fact" and doc.raw_key == target:
                return True
    return False


# ---------------- baseline: stock scorer logic (spec 11.3) ----------------

def _baseline_relevance_tokens(core, value):
    return core.content_tokens(value)


def _baseline_score(core, doc, query_tokens, now):
    """Approximates the stock core: token-overlap boost + confidence then recency."""
    value_tokens = set(_baseline_relevance_tokens(core, doc.text))
    overlap = len(set(query_tokens) & value_tokens)
    recency_days = max(0.0, (now - doc.ts) / 86_400.0)
    return (
        100.0 * doc.confidence
        + 10.0 * min(0.3, overlap * 0.1)
        - min(20.0, recency_days * 0.02)
    )


def _baseline_rank(core, docs, query_text, now, k=12):
    query_tokens = core.content_tokens(query_text)
    scored = [(_baseline_score(core, doc, query_tokens, now), doc) for doc in docs]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(doc, None) for _s, doc in scored[:k]]


# ---------------- metrics ----------------

def _metrics(core, docs, queries, rank_fn, k_values=(5, 10)):
    per_group = {}
    for query in queries:
        group = query["group"]
        ranked = rank_fn(core, docs, query)
        # rank_fn may return bare docs (baseline) or (doc, score) tuples.
        rows = [
            (doc_score[0], doc_score[1]) if isinstance(doc_score, tuple) else (doc_score, None)
            for doc_score in ranked
        ]
        relevant = _relevant_for_query(query)
        if group == "negative":
            # Relevant = none; count non-anchor FACTS that would actually be injected.
            non_anchor_hits = 0
            for pos, (doc, score) in enumerate(rows[: k_values[1]]):
                in_range = (score is None and pos < 12) or (score is not None and score >= core.MIN_SCORE)
                if in_range and not core.is_anchor(doc) and doc.kind == "fact":
                    non_anchor_hits += 1
            stats = per_group.setdefault(group, {"hits": 0, "count": 0, "recip": []})
            stats["hits"] += 0 if non_anchor_hits == 0 else 1  # 1 = failure
            stats["count"] += 1
            continue
        if not relevant:
            continue
        stats = per_group.setdefault(group, {"hits@5": 0, "hits@10": 0, "recip": [], "count": 0})
        found_5 = any(_is_hit(doc, relevant) for doc, _s in rows[:5])
        found_10 = any(_is_hit(doc, relevant) for doc, _s in rows[:10])
        recip = 0.0
        for pos, (doc, _score) in enumerate(rows, 1):
            if _is_hit(doc, relevant):
                recip = 1.0 / pos
                break
        stats["hits@5"] += 1 if found_5 else 0
        stats["hits@10"] += 1 if found_10 else 0
        stats["recip"].append(recip)
        stats["count"] += 1
    out = {}
    for group, stats in per_group.items():
        n = max(1, stats["count"])
        if group == "negative":
            out[group] = {
                "recall@5": float("nan"),
                "recall@10": float("nan"),
                "mrr": float("nan"),
                "negative_ok_rate": 1.0 - stats["hits"] / n,
                "count": n,
            }
        else:
            out[group] = {
                "recall@5": stats["hits@5"] / n,
                "recall@10": stats["hits@10"] / n,
                "mrr": sum(stats["recip"]) / n,
                "count": n,
            }
    return out


def _print_markdown_table(title, rows):
    print(f"\n### {title}\n")
    print("| group | n | Recall@5 | Recall@10 | MRR |")
    print("|---|---|---|---|---|")
    for name, metrics in rows:
        print(
            f"| {name} | {metrics['count']} | {metrics['recall@5']:.3f} | {metrics['recall@10']:.3f} | {metrics['mrr']:.3f} |"
        )


# ---------------- the evaluation ----------------

@pytest.fixture(scope="module")
def eval_setup(core):
    fixture = _load_fixture()
    retriever = core.get_retriever()
    docs = _all_docs(core, fixture)
    index = retriever.build_index(docs)
    return fixture, docs, index


def _scorer_rank(core, docs, query, index=None):
    retriever = core.get_retriever()
    hour = 20  # representative evening-hour turn
    results = retriever.rank(
        {"text": query["text"], "context_text": query.get("context_text", ""), "now": NOW, "hour": hour},
        index,
        k=12,
    )
    return [(item["doc"], item["score"]) for item in results]


def test_retrieval_targets(eval_setup, core):
    fixture, docs, index = eval_setup
    queries = fixture["queries"]

    ours = _metrics(core, docs, queries, lambda c, d, q: _scorer_rank(c, d, q, index))
    baseline = _metrics(core, docs, queries, lambda c, d, q: _baseline_rank(c, d, q["text"], NOW))

    print("\n===== RETRIEVAL EVALUATION (fixture person, grouped queries) =====")
    _print_markdown_table("New scorer (LexicalRetriever)", sorted(ours.items()))
    _print_markdown_table("Baseline (stock-style scorer)", sorted(baseline.items()))

    # --- spec 11.3 pass criteria ---
    for group, metrics in ours.items():
        if group == "negative":
            continue
        same_group = baseline.get(group)
        assert metrics["recall@5"] >= same_group["recall@5"] - 0.05, (
            f"{group}: recall@5 {metrics['recall@5']:.2f} below baseline {same_group['recall@5']:.2f}"
        )
        assert metrics["mrr"] >= same_group["mrr"] - 0.08, (
            f"{group}: mrr {metrics['mrr']:.2f} below baseline {same_group['mrr']:.2f}"
        )
    assert ours["paraphrase"]["recall@5"] >= 0.80, f"paraphrase recall@5 = {ours['paraphrase']['recall@5']:.2f}"
    assert ours["morphological"]["recall@5"] >= 0.80, f"morphological recall@5 = {ours['morphological']['recall@5']:.2f}"
    assert ours["negative"]["negative_ok_rate"] >= 0.8, f"negative group leaked in {1 - ours['negative']['negative_ok_rate']:.2f}"

    # exact must not regress versus baseline
    assert ours["exact"]["recall@5"] >= baseline["exact"]["recall@5"] - 0.05


def test_injection_selection_min_score(eval_setup, core):
    """Unrelated (negative) queries should surface only anchor-grade docs at selection time."""
    fixture, docs, index = eval_setup
    negative_queries = [q for q in fixture["queries"] if q["group"] == "negative"]
    retriever = core.get_retriever()
    leaks = 0
    for query in negative_queries:
        results = retriever.rank({"text": query["text"], "now": NOW}, index, k=12)
        non_anchor_above_min = [
            item for item in results if item["score"] >= core.MIN_SCORE and not core.is_anchor(item["doc"]) and item["doc"].kind == "fact"
        ]
        if non_anchor_above_min:
            leaks += 1
            print(f"  leak: '{query['text']}' ->", [(item["doc"].raw_key, round(item["score"], 3)) for item in non_anchor_above_min[:3]])
    assert leaks <= 2, f"{leaks}/10 negative queries surfaced non-anchor facts above MIN_SCORE"


def test_episodes_rank_for_episode_queries(eval_setup, core):
    fixture, docs, index = eval_setup
    episode_queries = [q for q in fixture["queries"] if q["group"] == "episode"]
    retriever = core.get_retriever()
    hits = 0
    for query in episode_queries:
        results = retriever.rank({"text": query["text"], "now": NOW}, index, k=5)
        kinds = [(item["doc"].kind, item["doc"].raw_text) for item in results]
        found = any(kind == "episode" and text in query["relevant_episode_texts"] for kind, text in kinds)
        hits += 1 if found else 0
    print(f"\nepisode queries: {hits}/{len(episode_queries)} recall@5")
    assert hits / len(episode_queries) >= 0.6
