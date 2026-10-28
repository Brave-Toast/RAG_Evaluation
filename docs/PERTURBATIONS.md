# Perturbation experiments

Select one or more methods in the dashboard. **Combined** applies them together to one
variant. **Separate comparisons** creates one independent variant for each selected
method. Both modes generate one clean baseline per request and reuse it for every
comparison. Select a result from the comparison dropdown to inspect its context,
answer, metrics, warnings, and changes.

## Methods and settings

| API method | Stage | Control | Behaviour |
|---|---|---|---|
| `spelling` | Before retrieval | `typo_rate`: 0–1, default 0.25 | Existing random character swaps, deletions, and substitutions; 0 preserves the question exactly. |
| `rephrase` | Before retrieval | On/off | Existing model rephrase with a labelled rule-based fallback. |
| `emoji` | Before retrieval | `emoji_style`, `emoji_count`, `emoji_position` | Insert emojis or encode words/numbers/dates using emoji, letters, and digits; record each insertion and replacement. |
| `negation_flip` | Before retrieval | On/off | Flips the question's polarity: removes an existing negation (`not`, `n't`), or inserts `not` after the first auxiliary verb (is, was, did, can, ...). Skipped if the question has neither. |
| `entity_substitution` | Before retrieval | On/off | Replaces one named entity in the question (a known candidate-document title, chosen with the seed) with the title of an unrelated benchmark question. Records the original and replacement entity. Skipped if no entity is found. |
| `metadata_injection` | Before retrieval | `metadata_term_count`: 1–10, default 4 | Adds one off-topic passage to the candidates with query-derived topic keywords prepended, so it ranks highly without containing the answer (see below). |
| `conflict` | Before retrieval | On/off | Adds a synthetic contradictory candidate document that asserts a wrong answer and carries a high score; retrieval may or may not select it. |
| `remove_evidence` | After retrieval | `remove_count`: 1–5, default 1 | Remove retrieved supporting passages if annotations exist; otherwise remove the highest-ranked passages and explicitly label the fallback. |
| `alter_facts` | After retrieval | `fact_count`: 1–20, default 1 | Replace sampled integer tokens, including years and date components, with a different integer. Record each old/new value and its position. Names and decimal values are not edited. |
| `truncate` | After retrieval | `retain_percent`: 1–100, default 50 | Keep a prefix of each passage up to this percentage of its characters, ending at a word boundary. A short passage can become empty. |
| `irrelevant` | After retrieval | `irrelevant_count`: 1–5, default 1 | Append sampled passages from other benchmark questions, excluding original candidate titles/text and duplicate distractors. Irrelevance is not guaranteed. |
| `shuffle` | After retrieval | On/off | Change passage order without editing text. Fewer than two distinct passages produces a reported no-op. |

The five post-retrieval methods (`remove_evidence`, `alter_facts`, `truncate`, `irrelevant`, `shuffle`) modify only the context supplied to generation. They do not
rebuild the persisted database or change retrieval algorithms. Edited passages
have no relevance score attached: their original retrieval score remains available
in the separately recorded retrieval output.

The bundled benchmark has no supporting-document annotations. No dataset changes
are needed: the fallback is explicitly reported as **highest-ranked passages
(support annotations unavailable)**. API callers may supply `supporting_titles`.
If annotations exist but no supporting passage was retrieved, removal is skipped;
it does not silently remove a different passage. Available annotated dataset fields
are also read (`supporting_titles`, or `supporting_facts.title`).

Removal/replacement/addition counts are capped by available applicable content;
the change summary reports requested and actual counts. Every selected method gets
an applied/skipped record. Removing all passages is allowed and visible.

## Composition and repeatability

Combined execution always follows this order, regardless of checkbox order:

```text
Rephrase -> spelling -> emoji -> negation flip -> entity substitution
-> metadata injection -> conflicting candidate -> retrieve -> [rerank, if enabled]
-> remove evidence -> alter facts -> truncate -> add distractors -> shuffle
-> generate answer
```

Distractors are added after evidence editing so their original content remains
intact. The final shuffle covers all remaining and added passages. Separate
comparisons start each variant with independent copies of the original inputs;
post-retrieval-only variants reuse an exact copy of the baseline retrieval output.

An optional `seed` (0–2147483647) controls random transformations. An omitted seed
is generated and returned in the result, history, and logs. Each method has its
own random stream. The same seed and same inputs reproduce its random choices;
earlier combined methods may change those inputs. Seeds do not control model
responses, embeddings, or external services.

## Negation flip, entity substitution, and metadata injection

These three pre-retrieval methods were added after the original nine. Together with the
original methods there are **12** perturbations: seven before retrieval (`rephrase`,
`spelling`, `emoji`, `negation_flip`, `entity_substitution`, `metadata_injection`,
`conflict`) and five after (`remove_evidence`, `alter_facts`, `truncate`, `irrelevant`,
`shuffle`).

**Negation flip** changes only the question text. It removes the first `not` or `n't`
if one exists; otherwise it inserts `not` after the first auxiliary verb. The change
record shows whether the question was flipped. Questions with no negation and no
auxiliary verb are reported as skipped. The flipped question may be ungrammatical or
no longer answerable; the ground truth is not changed.

**Entity substitution** looks for a candidate-document title that appears in the
question and swaps the first occurrence for the title of a document from another
benchmark question. The choice of entity and replacement is seeded. The change record
lists the original and replacement entity. If no title appears in the question, the
method is reported as skipped. The question then names an entity that the retrieved documents do not describe, so a
correct answer may come from the model's own knowledge instead of the context.

**Metadata injection** targets the retrieval stage. It picks one off-topic passage from
another benchmark question (preferring ones that share words with the query, then
choosing randomly among the top five using the seed), keeps its first sentence, and
prepends topic tags built from the question's most salient terms
(`metadata_term_count`, default 4). The passage is added to the candidates with the
title `Synthetic_MetadataBoosted: ...`. It contains no real answer content, so if it
reaches the model the answer can only be harmed. The change record lists the terms and
the source passage. The method is skipped when no salient terms or no distinct
distractor passages are available. This perturbation probes retrieval and reranking:
keyword-stuffed passages can outrank the real evidence, and the `contradiction_filter`
repair is designed to detect them.

## Emoji words, numbers, and dates

Emoji perturbation modifies the question before retrieval
and works in combined or separate mode. In combined mode it runs after spelling
errors so the typo method cannot split a newly generated emoji sequence.

| Setting | Values | Default |
|---|---|---|
| `emoji_style` | `insert` (insert symbols), `substitute` (encode words/dates), `mixed` (both) | `mixed` |
| `emoji_count` | Maximum 1–20 edits | 3 |
| `emoji_position` | `prefix`, `suffix`, `between_words` (insertion only) | `suffix` |

Encoding examples:

- `before` → `b4️⃣`, `great` → `gr8️⃣`, `first` → `1️⃣st`, `today` → `2️⃣day`.
- `be` → `🐝`, `books` → `📚` (a finite dictionary of common words).
- Other eligible words can replace one of the letters A/B/I/M/O/P with an emoji
  letter; for example, `magazine` can become `m🅰️gazine`. The chosen letter is seeded.
- `10/09/2026` → `1️⃣0️⃣/0️⃣9️⃣/2️⃣0️⃣2️⃣6️⃣`. Numeric encoding preserves digits,
  separators, and leading zeros; it does not change the date's value or reorder it.

One replaced word or numeric span (including a whole numeric date) counts as one
edit, even if its encoding contains several emojis. One inserted emoji also counts
as one edit. Mixed mode spends up to half the budget on replacements (at least one),
then uses the remaining budget for insertions. With an edit budget of one, mixed
mode encodes one eligible token, or inserts one emoji if no token can be encoded.

Replacement candidates are complete English words and numeric spans. Alphanumeric
IDs, existing keycaps, and words without a supported encoding are left intact.
Substitution-only mode reports a skipped operation when no eligible tokens exist;
mixed mode can still insert symbols. Empty questions are left unchanged. Insertion
between words uses token boundaries; a single-token question falls back to the end,
with that fallback recorded. Existing spaces and emoji sequences are preserved.

The audit records input/output questions, original/replacement strings, edit kinds,
and offsets in Unicode code points relative to the stated stage (`input_query` for
encoding, `after_encoding` for insertion). Seeds make the choices repeatable.
Word/letter encodings are intended readings, not verified semantic equivalents;
they can change tone, case, or interpretation. Inspect the transformed question
before interpreting the ground-truth comparison. Synthetic generation is still
scripted; use a working model to study answer behaviour.

To verify it in the dashboard, select **Emoji perturbation**, choose **Encode words
and dates**, and inspect **Perturbed Query / Trigger** plus **Changes and skipped
methods**. For a numeric-date experiment, use a question/document/ground-truth set
that actually contains the date; a custom query in the dashboard still uses the
selected benchmark's documents and answer.

API settings example (added to the normal evaluation request):

```json
{
  "perturbation_types": ["emoji"],
  "settings": {"emoji_style": "substitute", "emoji_count": 3},
  "seed": 42
}
```

## Reading results

- `baseline_contexts` and `perturbed_contexts` are the final generation inputs.
- `baseline_retrieved_contexts` and `perturbed_retrieved_contexts` preserve retrieval
  output before post-retrieval transformations.
- `retrieval_rank_shift_jaccard` compares retrieved document titles before edits;
  it measures set overlap, not order. It stays 1 for post-retrieval-only variants.
- `context_construction_loss` and the legacy `retrieval_sim` field measure final
  context cosine similarity. Higher means more similar despite the legacy names.
- Generation similarity, ground-truth similarity, amplification, and classification
  retain their existing formulas. Amplification divides answer change by final
  context change, and can be large when context change is close to zero.
- `changes` includes stage, method, applied/skipped status, summary, and details.
- Generation-source fields and warnings identify synthetic/scripted answers and
  failed provider calls. Synthetic answers use ground truth and do not simulate
  a real model's response to context or emoji edits. Rule-based rephrases are also
  labelled; equivalent meaning is not guaranteed.

Use a working generation model to study answer behaviour. Review the answer and
evidence directly; similarity scores alone do not establish factual correctness.
Existing custom-query behaviour is preserved: the dashboard still uses the
selected benchmark's documents and ground truth.

## API

`POST /api/run-evaluation` continues to accept the old `perturbation_type` field.
It now also accepts `perturbation_types` (up to 12 methods), `mode`, `settings`, `seed`,
and optional `supporting_titles`. When a list is supplied it takes precedence over the legacy
field. Duplicate selections are collapsed into the fixed application order.

Example request (replace the documents and ground truth for your experiment):

```json
{
  "baseline_query": "Which magazine started first?",
  "ground_truth": "First",
  "contexts": [
    {"title": "First", "text": "First magazine started in 1844."},
    {"title": "Second", "text": "Second magazine started in 1989."}
  ],
  "perturbation_types": ["alter_facts", "truncate", "shuffle"],
  "mode": "separate",
  "settings": {"fact_count": 1, "retain_percent": 50},
  "seed": 42,
  "top_k": 2,
  "llm_model": "synthetic-evaluator"
}
```

The response contains `evaluation_id`, `mode`, `selected_methods`, effective
`settings` (including `typo_rate`), `seed`, and `variants`. Each variant has a unique
`variant_id` and all comparison fields. Top-level comparison fields mirror the
first variant so existing single-method clients still work. Secrets are excluded
from history and stage logs.

`POST /api/run-evaluation/stream` accepts the same request and returns newline-delimited
JSON events: `progress`, `baseline`, `variant`, and `complete`. Progress describes
the actual baseline/variant being processed; `complete.result` contains the same
response as the regular endpoint. Validation errors return HTTP 422. Failures
after streaming starts produce an `error` event without a completion event.

### Repairs, reranker, and judge fields

The same request can also include:

| Field | Meaning |
|---|---|
| `repair_methods` | Registered repair methods to apply (see `GET /api/repairs`). Query repairs such as `fix_spelling` run before retrieval; the others run on the retrieved context before generation. |
| `use_reranker` | `true` adds a cross-encoder reranking stage after retrieval. Default `false`, so older runs behave as before. |
| `rerank_fetch_k` | Candidates the retriever fetches before the reranker keeps `top_k` (1–20, default 10). |
| `reranker_model` | One of `GET /api/rerankers`: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base` (default), `BAAI/bge-reranker-v2-m3`. Unknown names return HTTP 422. |
| `use_llm_judge` / `judge_model` | Enables LLM-as-judge scoring (correctness, faithfulness, context relevance) with a model from `GET /api/judges`. |

When the reranker is on, the failure-localisation swap test can blame the `RERANKING`
stage: it does so when the retriever's own top-`k` would have produced a correct answer.
Each run's reranker settings and pool sizes are logged in the `rerank_stage` record of
`backend/stage_run_logs.jsonl`.

## History, exports, and validation

The browser keeps the existing localStorage key and can load older records.
New records store one entry per variant, linked by the shared evaluation ID, with
settings, selected methods, seed, provider settings, and full comparison result.
CSV includes these fields and a quoted `Result JSON` column; JSON preserves the
structured record. `backend/stage_run_logs.jsonl` records each completed variant,
including retrieval output and final generation context separately.

Run focused offline tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

or, with pytest installed:

```powershell
python -m pytest tests -v
```

Tests isolate model loading and external generation, exercise actual perturbations
and orchestration, and verify API compatibility, request validation, streaming,
baseline reuse, variant isolation, seeded edits, context-stage separation, logs,
and empty/inapplicable inputs. No additional production dependencies are introduced.

An optional real-browser check is available with `python tests/browser_smoke.py`.
It requires Chrome (or a `CHROME_BINARY` path), `websocket-client`, cached MPNet
weights, and free ports 8017 and 9337. It runs single, combined, and separate
experiments using real embeddings/retrieval and synthetic generation, checks history,
CSV and streamed responses in the browser, and saves temporary diagnostic artifacts.
It uses an isolated browser profile and stops its own server and browser afterward.
