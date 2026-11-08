# Tater Advanced Memory

**Advanced Memory** (`advanced_memory`, module `advanced_memory_core`) is an enhanced replacement for Tater's stock Memory Core.
It is a fork of the stock core ([`Tater_Shop/cores/memory_core.py`](https://github.com/TaterTotterson/Tater_Shop))
that keeps the same architecture — background extraction from conversation history, People
identity integration, episodic memory, profile cards — and upgrades six things:

1. **Smarter retrieval without embeddings.** A pure-Python lexical scorer (BM25 + character
   trigrams + a built-in concept map + intent/time-of-day priors + quality prior, combined with
   pseudo-relevance feedback and MMR diversity) decides which facts and episodes to inject for
   the current turn. No vector database, no network, ~ms-scale scoring.
2. **Correctable facts.** Stock's confidence ratchet can make a stale fact nearly impossible to
   replace. Advanced Memory implements a `pending_alt` rule: a differing value replaces the stored one
   on correction, after two sightings of the same alternative, when confidence is close, or
   when the fact is old — keeping the previous value in a `previous` history (max 3).
3. **First-run backfill & import.** New scopes extract a bounded history window on first
   encounter, and existing stock Memory Core data (`mem:user:*`, `mem:room:*`, identity docs)
   can be imported read-only and idempotently from the Management tab.
4. **Safer injection & maintenance UI.** An injection sanitizer drops secret-bearing and
   instruction-like values before they reach the model, sensitive categories are gated by
   default, LLM-proposed nightly consolidation runs behind all-or-nothing validation with
   rotating backups, and a **Recall Tester** in the WebUI shows exactly what would be injected
   and why (per-signal score breakdown).
5. **Inferred habits (opt-in).** With *Inferred Habits* enabled, a nightly, zero-LLM pass mines
   other cores' read-only history — Awareness events (`tater:automations:events:*`), the
   Automation-GA core's entity journal (`automation_ga_core:journal`), **per-person listening
   histories from the Personal Music Core** (`personal_music_core:history:v1:<person_id>`),
   the stock Music Core's listening history (`music_core_listening_history_v1`),
   Awareness **Face-ID presence sessions** (`awareness:face_session:*` — who is usually in
   which room at which hour, with weekday/weekend skews), and recurring request categories
   in your own chat histories — and proposes time-of-day patterns in the **Habits** tab.
   Nothing is stored for the model until you **Approve** it (it becomes an inferred fact
   injected with a "(likely)" marker); **Dismiss** records the pattern so it is never
   re-proposed. A quiet home produces no proposals and costs nothing.
6. **Automation suggestions from approved routines.** Approved habit patterns are tracked in a
   recurrence registry; once a pattern has recurred enough more times (setting, default 5),
   Advanced Memory offers to create a matching automation in your automation core — visible in
   the Habits tab, and optionally proactively pushed via a **notification** and/or a **TTS
   announcement** (either/or/both per user setting). Announcements route to the voice satellite
   assigned to the room where the routine happens (e.g. the gym SAT for a 5am gym routine),
   falling back to a user-configured default satellite. Approving a suggestion hands the request
   **through Hydra**, which authors the automation in the automation core under its own
   validation/review gates — nothing runs until approved there. Scope setting lets you decide
   which patterns qualify: paired presence+action routines (`composite`, default), only
   person-attributed routines (`person_routines`), or any approved pattern (`any_pattern`).

## Install

1. Add the raw manifest URL of this repository to Tater under **Cores → Repositories**:
   `https://raw.githubusercontent.com/heapsoftware/Tater-Advanced-Memory/main/core_manifest.json`
2. Install **Advanced Memory** from the Shop list and configure it under its settings category
   (**Advanced Memory Settings**).
3. Start the core. Facts appear in the **Adv Memory** tab as conversation history is processed.

> ⚠️ Tater extensions run **in-process and unsandboxed** inside your Tater installation. They
> have access to Tater's Redis, filesystem, and network. Only install code from repositories
> you trust. Advanced Memory stores all data in Tater's own Redis under the `advanced_memory_core:*`
> namespace and speaks to your configured LLM provider; it does not send data anywhere else.

## Coexisting with the stock Memory Core (A/B testing)

Advanced Memory can run **alongside** the stock core without disturbing it:

- **Shadow mode** (default path for evaluation): leave the stock core running and set
  **Inject Memory Context OFF** in Advanced Memory settings. Advanced Memory still extracts, stores, and
  consolidates, but injects nothing and does not claim `memory_add`/`memory_remove`, so the
  stock core remains the injected memory and tool owner.
- Use the **Recall Tester** tab against real queries to compare with what the stock core
  injects; tune `custom_synonyms` (one group per line, comma-separated words) to cover your
  vocabulary.
- When the Overview tab shows the stock core running **and** injection ON, a conflict warning
  appears there and is logged (once per hour) — both cores would inject and fight over tools.
- **Cutover:** disable the stock Memory Core, set **Inject Memory Context ON**. Use
  **Import From Stock Memory Core** in Maintenance first if you want your existing facts.

### Manual test checklist

1. Install; shadow mode; let extraction + backfill run for a day.
2. Recall Tester with real queries; compare injected block against stock.
3. Verify injected block stays within the `context_max_chars` limit (shown in tester output).
4. Cutover; exercise `memory_add` / `memory_remove` / `memory_recall` by voice and WebUI.
5. State a fact, contradict it later; confirm the new value wins and the old value is kept in
   `previous` (visible after a forget/edit cycle or in the facts table).
6. Enable *Inferred Habits* after a few days of smart-home activity; review the Habits tab,
   approve one pattern, and confirm it injects with the "(likely)" marker; dismiss another and
   confirm it never re-appears.
7. Lower **Offer automations after recurrences** to 2 with **Automation suggestion channel =
   notification**; once an approved pattern recurs, confirm the Habits tab shows an
   "Automation:" suggestion, a notification arrives, and approving it hands the request to
   Hydra (the automation core's review still applies). Dismissing the idea must prevent any
   repeat offer.
8. Stop/restart the core — shutdown is cooperative and completes within seconds.
9. Uninstall with "Delete data" — only `advanced_memory_core:*` keys and the settings hash are
   removed; shared Tater label keys (`tater:room_label:*`, `tater:user_label:*`) are preserved.

## Retrieval quality (no-embeddings evaluation)

The scorer is evaluated against a fixture person (62 facts + 40 episodes, 84 labeled queries)
with a stock-style baseline (token overlap + confidence/recency). Run
`python3 -m pytest tests/test_evaluation.py -q -s` to reproduce the current table:

| group | n | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| episode | 10 | 0.800 | 0.900 | 0.594 |
| exact | 15 | 1.000 | 1.000 | 0.900 |
| morphological | 10 | 0.900 | 0.900 | 0.717 |
| paraphrase | 19 | 0.842 | 1.000 | 0.636 |
| short_utterance | 10 | 0.900 | 1.000 | 0.671 |
| typo | 10 | 0.900 | 0.900 | 0.783 |
| negative | 10 | 10/10 inject nothing beyond anchors | — | — |

(Baseline stock-style scorer for the same groups: episode 0.00, exact 0.33, morphological 0.10,
paraphrase 0.21, short_utterance 0.10, typo 0.00 Recall@5.)

Benchmark on this fixture: index build ≤ 250 ms and ranking ≤ 75 ms per query for 500 facts +
200 episodes (asserted in CI).

## Known limitations

- **Lexical retrieval is not semantic search.** Queries that share no words *and* no concept
  with the stored text will not match. Mitigations built in: pseudo-relevance feedback,
  character-trigram fuzziness, the built-in concept map (~90 groups), user-editable
  `custom_synonyms`, and the `memory_recall` tool (the planner can rephrase and retry).
- The concept map is English-oriented and hand-curated; rare hobbies/vocabulary should be
  added via `custom_synonyms`.
- Episodes and profile cards depend on the extraction LLM following the schema; malformed
  output is dropped, never guessed.
- Advanced Memory is intentionally a fork of the stock schema, so `Import From Stock Memory Core`
  only copies user/room facts and identity docs — it does not migrate stock retrieval caches.

## Development

```bash
python3 -m py_compile cores/advanced_memory_core.py
python3 -m pytest tests/ -q          # 99 tests incl. the evaluation harness
python3 reference/Tater_Shop/tools/generate_core_manifest.py --root . --name "Tater Advanced Memory"
python3 reference/Tater_Shop/tools/validate_package.py --root . --kind core
```

The manifest (`core_manifest.json`) is generated — do not hand-edit it. Regenerate it after
every source change and bump `__version__` in `cores/advanced_memory_core.py` whenever users
should receive an update.
