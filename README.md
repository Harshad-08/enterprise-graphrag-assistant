# Enterprise Knowledge Assistant: Reliability-Focused GraphRAG

A question-answering assistant over company policies, standards and procedures. It combines a **Neo4j knowledge graph** with **hybrid semantic and keyword retrieval**, and focuses on three reliability problems:

1. **A relevant document exists but was not retrieved.** Evidence sufficiency is assessed, then bounded graph-based recovery runs.
2. **Documents contradict each other.** Contradictions are detected claim by claim and resolved by explicit precedence rules, or disclosed when they can't be resolved.
3. **The LLM makes unsupported statements.** Each claim is checked against its evidence, citations are validated, there is one bounded correction attempt, and the system abstains when nothing verifiable remains.

The repository includes a synthetic enterprise corpus, a labelled evaluation set, a baseline-vs-improved evaluation (custom metrics plus Ragas), a FastAPI backend, a Streamlit UI and a pytest suite.

> This is a portfolio/learning project. It has **not** had the security, scale and operational validation that a production deployment needs.

---

## Screenshots

Captured from the running system: Streamlit UI and FastAPI, Neo4j 5.26, local embedding/reranker/NLI models, and Groq as the LLM. The sidebar in each image shows the model that produced the answer. Most chat screenshots used `openai/gpt-oss-20b` because `gpt-oss-120b` had reached its free-tier daily token limit; screenshot 4 used `gpt-oss-120b`.

| | |
|---|---|
| **Resolved conflict.** v2.0 supersedes v1.0; the overridden rule is shown, not used.<br>![Resolved conflict](docs/screenshots/01_chat_resolved_conflict.png) | **Graph recovery.** "Atlas" → `RELATED_TO` → Production Database; relationship and rules cited separately, every claim verified.<br>![Graph recovery](docs/screenshots/02_chat_graph_recovery.png) |
| **Unresolved conflict.** Equal authority and a missing effective date, so both rules are disclosed.<br>![Unresolved conflict](docs/screenshots/03_chat_unresolved_conflict.png) | **Multi-hop question.** Answered from the `DEPENDS_ON` edge, citing the procedure document.<br>![Multi-hop](docs/screenshots/04_chat_multi_hop.png) |
| **Abstention.** The topic is absent from the corpus, which is concluded only after recovery.<br>![Abstention](docs/screenshots/05_chat_abstention.png) | **Document management.** Upload with metadata; indexed documents with status and authority.<br>![Documents](docs/screenshots/06_documents.png) |
| **Evaluation dashboard.** Baseline vs improved, corpus-level conflict metrics, matched Ragas comparison.<br>![Evaluation](docs/screenshots/07_evaluation.png) | **REST API.** FastAPI OpenAPI docs.<br>![API docs](docs/screenshots/08_api_docs.png) |

## Contents
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [How each challenge is solved](#how-each-challenge-is-solved)
- [Knowledge graph schema](#knowledge-graph-schema)
- [Setup and running](#setup-and-running)
- [API](#api)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)

---

## Architecture

```mermaid
flowchart LR
  U[Upload PDF/DOCX/TXT] --> P[Parse + validate] --> M[Metadata + chunking]
  M --> X[Entities, relations, claims]
  X --> G[(Neo4j graph + vector index)]
  M --> B[BM25 index]
  X --> C[Contradiction detection + precedence] --> G

  Q[Question] --> A[Query analysis: entities + concepts]
  A --> H[Hybrid retrieval: vector + BM25 -> RRF -> cross-encoder]
  H --> S{Evidence sufficient?}
  S -- no --> R[Graph recovery: traverse, expand, reformulate, re-rank] --> S
  S -- still no --> AB[Abstain with reason]
  S -- yes --> K[Conflict analysis over evidence + graph]
  K --> L[Evidence-constrained generation, Groq / Gemini / mock]
  L --> V[Claim verification + citation validation]
  V -- unsupported --> CO[One correction attempt] --> V
  V --> F[Answer + citations + conflict disclosure, or abstain]
```

Each stage is its own module with typed inputs and outputs (see [Project structure](#project-structure)):

| Stage | Module |
|---|---|
| Upload validation, filename sanitisation, size limits | `app/services/document_service.py`, `app/api/routes_documents.py` |
| PDF (PyMuPDF), DOCX (python-docx), TXT parsing | `app/ingestion/parsers.py` |
| Front-matter metadata, chunking with stable IDs, pages, sections | `app/ingestion/metadata.py`, `app/ingestion/chunking.py` |
| Entity linking, relationship extraction | `app/graph/entity_extraction.py`, `app/graph/relationship_extraction.py` |
| Graph construction (Neo4j / in-memory) | `app/graph/neo4j_client.py`, `app/graph/memory_store.py`, `app/graph/graph_queries.py`, `app/graph/schema.py` |
| Embeddings and vector search, BM25, RRF, reranking | `app/retrieval/vector_search.py`, `keyword_search.py`, `fusion.py`, `reranking.py`, `hybrid.py` |
| Evidence sufficiency | `app/retrieval/evidence_assessment.py` |
| Graph retrieval and recovery | `app/retrieval/graph_retrieval.py`, `app/retrieval/recovery.py` |
| Claim extraction, contradiction detection, conflict resolution | `app/reliability/claim_extraction.py`, `contradiction_detection.py`, `conflict_resolution.py`, `conflict_analysis.py` |
| Generation (Groq / Gemini / mock) | `app/generation/llm_client.py`, `prompts.py`, `answer_generator.py` |
| Verification, citations, abstention | `app/reliability/answer_verification.py`, `citations.py`, `abstention.py`, `nli.py` |
| Orchestration | `app/services/chat_service.py` |
| Evaluation | `app/evaluation/*` |

---

## How each challenge is solved

### 1. A relevant document is missed by retrieval

**Initial retrieval** (`hybrid.py`): vector top-20 (Neo4j vector index, all-MiniLM-L6-v2) and BM25 top-20 are fused with **Reciprocal Rank Fusion**, then filtered by metadata (department, type, status) and reranked with **cross-encoder/ms-marco-MiniLM-L-6-v2**. Scores are mapped through a sigmoid so thresholds read as probabilities. Document and chunk IDs and all scores are kept on each result. For relationship questions ("which procedure *depends on* …"), graph traversal runs as a planned step.

**Sufficiency** (`evidence_assessment.py`). A high similarity score is never enough on its own. The decision combines:
- best cross-encoder relevance
- query-entity coverage, where graph-equivalent entities count: evidence about *Production Database* covers a question about *Atlas* because `Atlas -[RELATED_TO]-> Production Database`
- coverage of the question's non-entity concepts ("access", "retention"), using catalog synonyms
- whether a relevant passage contains an extracted claim about those entities and concepts

There are two hard gates: at least one passage must be relevant, and concept coverage must pass. The concept gate is waived only when the cross-encoder is very confident *and* all entities are covered. Thresholds are calibrated (see [Evaluation](#threshold-calibration)).

**Recovery** (`recovery.py`), bounded by `RECOVERY_MAX_ATTEMPTS` (default 2). Each round:
1. Links query entities to graph nodes.
2. Traverses GOVERNS / DEPENDS_ON / APPLIES_TO / OWNED_BY / RELATED_TO edges. The hop radius grows per round, and paths never pass *through* department hubs.
3. Collects passages from connected documents.
4. Expands the query with synonyms and related entity names, and **reformulates** it by substitution ("…for the Atlas cluster" becomes "…for the Production Database").
5. Re-runs hybrid retrieval.
6. Merges and reranks against the original question plus the reformulations.
7. Reassesses sufficiency.

**Distinguishing failure types.** After the last round the outcome is classified as one of:
- `recovered`: the document existed and was missed
- `documents_without_answer`: entities exist in the graph, but no passage answers the question
- `not_in_corpus`: no graph entity matches, and most query concepts never occur in the BM25 vocabulary
- `insufficient_after_recovery`: none of the above apply

"Absent" is only concluded after every recovery round has failed, never after a single search.

On this corpus, for *"What are the access requirements for the Atlas cluster?"*, the cross-encoder ranks the Production Database Access Standard near zero, because it doesn't know Atlas is a production database. Recovery follows `Atlas Cluster -[RELATED_TO]-> Production Database <-[GOVERNS]- Production Database Access Standard` and answers with verified citations to SEC-STD-004.

### 2. Documents contradict each other

**Claims** (`claim_extraction.py`). Normative sentences become structured claims with:
- subject entity and normalised property key
- polarity (require / permit / prohibit / state)
- quantity normalised to a base unit ("12 months" and "90 days" become minutes)
- applicability conditions ("only when …", "if …", "unless …")
- full provenance: document, chunk, page, section, publication and effective dates, version, status and authority

LLM-extracted claims (optional, `EXTRACTION_BACKEND=llm`, any configured provider) are kept only if their text appears verbatim in the source chunk.

**Detection** (`contradiction_detection.py`). Only claims about the same subject in *different* documents are compared:
- **Structured** (same property): opposing polarity, or different quantities in the same unit, is a contradiction *only if the conditions match*. Different conditions give `COMPATIBLE_CONDITIONAL`. For example, a general ban on personal devices and a webmail-only exception are compatible.
- **Semantic** (property unknown): embedding similarity first, then the NLI model must find contradiction in either direction. Without an NLI model these pairs are left unclassified rather than guessed.
- Different wording on its own never counts as a contradiction. Same-rule pairs are stored as `SUPPORTS`.

**Resolution** (`conflict_resolution.py`). Rules are applied in configurable order: `CONFLICT_PRECEDENCE=supersession,status,authority,effective_date,version`.

| Rule | Decides when |
|---|---|
| supersession | an explicit (transitive) SUPERSEDES edge exists; circular metadata is ignored |
| status | statuses are known and differ (active > draft > superseded/archived) |
| authority | both authorities are known and differ (`AUTHORITY_LEVELS`) |
| effective_date | one document is not yet effective, **or** both have known, *equal* authority, are active, and have different dates |
| version | versions of the same titled document |

A rule that lacks the metadata it needs is skipped, and the reason is recorded. If no rule decides, the conflict is **unresolved**. The answer then shows both claims verbatim with their citations and doesn't choose between them. "Newest wins" is never applied when authority is unclear.

At query time (`conflict_analysis.py`), claims in the evidence are matched against claims on the same subject and property *anywhere in the graph*. A contradicting document that retrieval missed is still detected, and its passage is added to the evidence. Answer claims that restate an overridden rule (same numbers, same polarity, high overlap) are removed.

### 3. Unsupported LLM answers

- **Constrained generation.** The LLM sees evidence labelled `E1…En` and must return structured `{claims: [{text, citation_ids}]}` JSON (strict JSON-schema output on Groq). It never sees or emits document IDs, chunk IDs or page numbers. The application builds citations from the labels.
- **Claim-level verification** (`answer_verification.py`). Each claim's citation labels must refer to passages that were actually supplied. Each cited passage is then checked with the configured verifier:
  - `nli`: local cross-encoder NLI over sentence windows
  - `llm`: an LLM judge (Groq or Gemini)
  - `lexical`: heuristic fallback

  A hard rule also applies: every number in a claim must appear in the supporting passage. Possible verdicts are `supported`, `unsupported`, `contradicted` and `invalid_citation`. A claim that another supplied passage supports gets its citation deterministically re-attributed.
- **Correction and abstention.** If any claim is unsupported, one correction call is made with the verifier's feedback (`MAX_CORRECTION_ATTEMPTS=1`) and the result is re-verified. Claims that still fail are removed. If nothing verified remains, the system abstains. Self-reported model confidence is never used.
- **Citation validation** (`citations.py`). Before returning, each citation is re-checked against the store: the chunk exists, the document ID matches, the excerpt is verbatim chunk text, and the page matches. Pages are only shown for PDFs, so none are fabricated.

Example (maternity leave is absent from the corpus): *"I couldn't verify information about maternity leave from the available documents. After retrieval recovery, no indexed document appears to cover this topic. Please consult the appropriate policy owner…"*

---

## Knowledge graph schema

Nodes: `Document`, `Chunk`, `Claim`, plus entities, which carry a generic `:Entity` label and one typed label: `Policy`, `Department`, `System`, `Procedure`, `Concept`. `Concept` was added for data categories and device types such as "Personal Devices" or "Application Logs", which are neither systems nor policies.

```
(Document)-[:CONTAINS]->(Chunk {embedding})          (Chunk)-[:MENTIONS]->(Entity)
(Document)-[:CONTAINS]->(Claim)-[:EXTRACTED_FROM]->(Chunk)
(Claim)-[:REFERS_TO]->(Entity)                       (Document)-[:DEFINES]->(Policy|Procedure)
(Policy)-[:GOVERNS]->(System)                        (Procedure|Policy)-[:DEPENDS_ON]->(Policy)
(Policy|Procedure|System)-[:OWNED_BY]->(Department)  (Policy)-[:APPLIES_TO]->(Concept|System)
(Entity)-[:RELATED_TO]->(Entity)                     (Document)-[:SUPERSEDES]->(Document)
(Claim)-[:CONTRADICTS|SUPPORTS|RELATED_TO {kind, method, resolved, winner_document_id, rule}]->(Claim)
```

- Uniqueness constraints on `document_id`, `chunk_id`, `claim_id` and `entity_id` give stable IDs and idempotent re-ingestion.
- There are indexes on entity name, chunk document, claim (subject, property), claim chunk and document status, plus a 384-dimension cosine **vector index** and a full-text index. See `app/graph/schema.py`.
- Entity-to-entity relationships store `source_documents` and `source_chunks` provenance. Deleting a document retracts its provenance, deletes relationships nobody else asserts, and removes orphaned entities.
- All Cypher is documented in `app/graph/graph_queries.py`: construction, deletion, vector search, traversal and provenance lookups.

---

## Setup and running

Tested with **Python 3.13** on macOS (arm64), Neo4j **5.26 Community** (Docker) and the pinned `requirements.txt`. The models download from Hugging Face on first use (about 200 MB in total).

### Option A: local Python + Neo4j in Docker (recommended for development)

```bash
cd enterprise-graphrag-assistant
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set NEO4J_PASSWORD; GROQ_API_KEY (or GEMINI_API_KEY) for a real LLM

docker compose up -d neo4j    # Neo4j Browser at http://localhost:7474
python scripts/initialize_database.py
python scripts/seed_synthetic_data.py

uvicorn app.main:app --port 8000                       # API docs: http://localhost:8000/docs
API_BASE_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py   # UI: http://localhost:8501
```

### Option B: everything in Docker Compose

```bash
cp .env.example .env    # set NEO4J_PASSWORD and GROQ_API_KEY (or GEMINI_API_KEY)
docker compose up --build
# API http://localhost:8000 (or API_PORT) · UI http://localhost:8501 · Neo4j http://localhost:7474
```

The API seeds the synthetic corpus on startup (`SEED_ON_STARTUP=true` in compose) and the models are baked into the image.

### With and without external credentials

| Mode | Configuration | What you get |
|---|---|---|
| Groq (tested) | `GROQ_API_KEY=...`, `LLM_PROVIDER=auto` or `groq`, `GROQ_MODEL=openai/gpt-oss-120b` (any chat model your key can access; check with the Groq console), `GROQ_REASONING_EFFORT=low` | Real generation and correction with strict JSON-schema output. Ragas evaluation uses `RAGAS_JUDGE_MODEL` (default `openai/gpt-oss-20b`, a separate per-model rate-limit bucket). |
| Gemini (implemented, not tested) | `GEMINI_API_KEY=...`, `LLM_PROVIDER=gemini`, `GEMINI_MODEL=gemini-2.5-flash` | Same features through the google-genai SDK. With `auto`, Gemini takes priority when both keys are set. |
| Mock (no key) | `LLM_PROVIDER=auto` with an empty key, or `mock` | A deterministic **extractive** mock copies evidence sentences that overlap the question. Every answer is prefixed `[MOCK MODE …]` and the UI shows a banner. Retrieval, recovery, conflict handling, verification (local NLI) and abstention all run for real. |
| Fully offline / no Neo4j | `GRAPH_BACKEND=memory EMBEDDING_BACKEND=hashing RERANKER_ENABLED=false VERIFIER_BACKEND=lexical SEED_ON_STARTUP=true` | In-process store and lexical-only components, with no downloads. For demos and tests only, and nothing persists. |

If `LLM_PROVIDER=groq` or `gemini` is set without that provider's key, startup fails loudly. The system never silently switches providers. If a model name is unavailable to your key (for example, the test key had no Llama models), the provider returns `model_not_found`; set `GROQ_MODEL` or `GEMINI_MODEL` to a model your account lists.

**Rate limits.** The free Groq tier used here allows 8,000 tokens per minute per model. The SDK retries HTTP 429 responses with backoff (honouring `Retry-After`, `LLM_MAX_RETRIES=8`), so a full evaluation run takes many minutes. gpt-oss reasoning tokens count towards the limit, which is why `GROQ_REASONING_EFFORT=low` is the default. Reasoning text is never shown or used; only the final JSON is parsed.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Backend, connectivity, model and verifier info, graph statistics |
| POST | `/documents/upload` | Multipart `file` (PDF/DOCX/TXT, ≤ `MAX_UPLOAD_MB`) plus optional metadata form fields (`title`, `document_id`, `doc_type`, `department`, `version`, `status`, `authority`, `effective_date`, `publication_date`, `supersedes`) |
| GET | `/documents` | Indexed documents with metadata, chunk and claim counts |
| GET | `/documents/{id}` | One document |
| DELETE | `/documents/{id}` | Removes graph nodes, vectors (stored on chunk nodes), claims, provenance, BM25 entries and the stored file |
| POST | `/chat` | `{"question": "...", "mode": "improved" \| "baseline", "top_k"?, "department"?, "doc_type"?}` |
| POST | `/evaluation/run` | `{"configurations": ["baseline","improved"], "include_ragas": false, "limit"?, "categories"?}` |
| GET | `/evaluation/latest` | Last saved report |

The chat response includes `answer`, `citations` (title, document ID, chunk ID, page and section when available, verbatim excerpt, status, version), `supporting_chunk_ids`, `recovery_performed`, `conflict_detected` plus `conflicts` (both claims, sources, resolution rule, explanation), `verified`, `abstained`, `retrieval_outcome`, `reliability_notes`, `graph_paths` and `claim_verifications`. No chain-of-thought is exposed; explanations are user-facing summaries of the decisions made.

---

## Evaluation

```bash
python scripts/run_evaluation.py            # baseline vs improved, custom metrics
python scripts/run_ragas.py                 # Ragas over the saved report (resumable; needs GROQ_API_KEY or GEMINI_API_KEY)
python scripts/run_evaluation.py --ragas    # both in one go (fine without tight rate limits)
python scripts/calibrate_thresholds.py      # re-calibrate sufficiency thresholds
```

Generation and judging are separate steps so a rate-limit stall while judging never loses the generated answers. `run_ragas.py` checkpoints every scored item in `data/evaluation/results/ragas_checkpoint.jsonl`, and re-running it resumes. On Groq, metrics are spread across judge models (`context_precision` uses `openai/gpt-oss-20b`; the others use `openai/gpt-oss-120b`; override with `RAGAS_JUDGE_MODEL` or `RAGAS_JUDGE_MODELS=metric:model,...`) because each model has its own rate-limit bucket. A metric always uses the same judge for both configurations.

Reports are written to `data/evaluation/results/` as `evaluation_<ts>.json` (full detail, including every item), `comparison_<ts>.csv` and `items_<ts>.csv`. The UI's Evaluation tab runs the same pipeline and exports JSON and CSV.

**Dataset** (`data/evaluation/eval_dataset.json`): 28 labelled questions. Categories: answerable (8), multi-hop (3), graph recovery (2), query expansion (1), resolved conflicts (4), unresolved conflict (1), compatible conditions (1), unanswerable (4) and unsupported-generation triggers (4). Corpus-level gold conflicts are in `gold_conflicts.json` (4 contradictions, 2 compatible pairs).

**Baseline** = plain top-k vector search plus one generation call, with no fusion, reranking, recovery, conflict handling, verification or abstention logic. **Improved** = the full pipeline. Both use the same questions, models and LLM.

### Metric definitions

| Metric | Definition (denominator) |
|---|---|
| retrieval_recall_at_k | mean over items with relevant docs of \|relevant ∩ top-K evidence docs\| / \|relevant\| |
| recovery_success_rate | answerable items where recovery ran and ended `recovered` with a relevant doc in the evidence / answerable items where recovery ran |
| expected_recovery_hit_rate | items labelled `expects_recovery` whose final evidence contains a relevant doc / those items |
| citation_validity_rate | citations referring to a supplied, stored passage / all citations in answers |
| claim_support_rate | supported claims / all claims in non-abstained answers (judged by the configured verifier) |
| unsupported_answer_rate | non-abstained answers with ≥1 unsupported claim / non-abstained answers |
| abstention_precision / recall | correct abstentions / all abstentions; correct abstentions / items labelled `should_abstain` |
| false_abstention_rate | abstentions on answerable items / answerable items |
| expected_fact_recall | expected fact strings found in the answer / expected facts (answerable items) |
| stale_answer_rate | answers stating a superseded or overridden fact without a conflict disclosure / items with stale facts |
| query_conflict_handling_accuracy | items where the response reports the expected resolved winner or an unresolved conflict / items with an expected conflict |
| contradiction precision / recall | corpus-level (document pair, subject) matches: TP / detected pairs; TP / gold pairs |
| conflict_resolution_accuracy | gold contradictions whose detected resolution matches (winner or unresolved) / all gold contradictions |

**Ragas** (0.4.3, `ragas.metrics.collections`): `ContextRecall`, `ContextPrecisionWithReference`, `Faithfulness` and `AnswerCorrectness`, with local HuggingFace embeddings. On Groq, the judge is an `openai.AsyncOpenAI` client pointed at Groq's OpenAI-compatible endpoint (`llm_factory(provider="openai")`), because ragas 0.4.3's native `provider="groq"` path calls an Anthropic-style `client.messages.create` that the Groq SDK lacks. On Gemini, it uses `llm_factory(provider="google")`. Context metrics and AnswerCorrectness run only on items that have a reference answer. Faithfulness is skipped for abstentions. Ragas measures quality; it does not implement any of the mechanisms above. Without a key the report says `skipped` and shows no scores.

### Results actually obtained (2026-10-24, Groq)

Environment: **Groq `openai/gpt-oss-120b`** (reasoning effort low) for both configurations, Neo4j 5.26, all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2 reranker, nli-deberta-v3-xsmall verifier, calibrated thresholds (0.75 / 0.5 / 0.5). The report is `data/evaluation/results/groq_generation_report.json`. 28 questions per configuration, 0 errors.

| Metric | Baseline | Improved |
|---|---|---|
| retrieval_recall_at_k | 0.950 | 1.000 |
| recovery triggered (count) | 0 | 11 |
| recovery_success_rate | n/a | 1.000 |
| citation_validity_rate | 1.000 | 1.000 |
| claim_support_rate (internal NLI verifier) | 0.460 | 1.000 |
| unsupported_answer_rate (internal NLI verifier) | 0.524 | 0.000 |
| abstention_precision | 1.000 | 1.000 |
| abstention_recall | 0.875 | 1.000 |
| false_abstention_rate | 0.000 | 0.000 |
| expected_fact_recall | 0.778 | 1.000 |
| stale_answer_rate | 0.200 | 0.000 |
| query_conflict_handling_accuracy | 0.000 | 1.000 |
| mean latency (ms, includes rate-limit retries) | 3,287 | 6,662 |

Corpus-level contradiction detection: precision 1.0, recall 1.0 (4/4), resolution accuracy 1.0, compatible pairs wrongly flagged: 0.

**Ragas: inconclusive (small matched sample).** Groq's free tier allows 200,000 tokens per day per model, and a full Ragas run needs roughly 10,000 judge tokens per item. The first attempt (gpt-oss-120b / gpt-oss-20b judges) used up both models' daily quota before scoring the improved configuration, so its partial numbers are not used as a comparison. A second run used a single judge for every metric and both configurations (`qwen/qwen3.8-27b`, reasoning off, `RAGAS_MAX_TOKENS=1000` because of qwen's 1,000 output-tokens-per-minute limit), interleaving the two configurations per question. It also hit the daily quota, after which only questions scored for **both** configurations are compared:

| Ragas metric (qwen judge, matched questions) | n | Baseline | Improved |
|---|---|---|---|
| context_recall | 3 | 0.344 | 0.544 |
| context_precision | 4 | 0.300 | 0.467 |
| faithfulness | 4 | 0.771 | 0.548 |
| answer_correctness | 2 | 0.494 | 0.842 |

With n = 2–4 these numbers support no conclusion. Faithfulness favoured the baseline for an identified reason: the improved answers end with deterministic conflict notes ("HR-POL-007 explicitly supersedes HR-POL-007-V1…") grounded in document **metadata**, but the judge was given only passage text. Q02–Q04 each scored 0.5: one verified claim plus one note. The baseline's Q04 (0.33) was penalised the same way for citing versions. **Fixed after this run:** evaluation contexts now contain every passage supplied to the generator (including conflict and graph-hop passages) with its metadata header, document ID and supersession, while the generator's prompt is unchanged. This has not been re-scored yet because the daily quota was used up. To finish:

```bash
python scripts/run_evaluation.py                       # regenerate answers with the judge-visible contexts
RAGAS_JUDGE_MODEL=qwen/qwen3.8-27b RAGAS_MAX_TOKENS=1000 python scripts/run_ragas.py --referenced-only
# re-run the same command after a quota reset to resume; --checkpoint-only aggregates without API calls
```

**Read these numbers with caution:**
- **The dataset is small and was written by the same author as the system, and some lexicon entries were added after seeing failures.** For example, "long" maps to durations and "valid" maps to "expire", added after two false abstentions appeared in development runs. This is test-set leakage, so the scores are optimistic. They demonstrate that the mechanisms work; they are not a benchmark.
- **The internal verifier is strict, which inflates the baseline gap on claim support.** Reading the baseline answers it marked unsupported, several are false negatives:
  - Q05: the claims are correct but begin "The Remote Access Standard specifies…". The document title is metadata, not passage text, so NLI does not confirm it.
  - Q13: "remain valid for 60 minutes" vs. "expire after 60 minutes" is a paraphrase the small NLI model misses.
  - Q26: "the documents do not specify…" is counted as a claim rather than as an abstention.

  Others are genuine: Q10 attributes production-database rules to Atlas without any passage saying so, and Q08 wrongly says no procedure exists. Ragas Faithfulness is meant to be the independent check, but see the inconclusive Ragas result above. Candidate fixes: include the document title and section in the verifier premise, use a larger NLI model (for example nli-deberta-v3-base), or use `VERIFIER_BACKEND=llm`.
- **The improved pipeline also had a run-to-run failure on Q10** during development: the LLM fused "Atlas" into each rule, every claim was rejected, and the system abstained, which is safe but unhelpful. The graph-relationship prompt rule and "keep the better attempt" correction were added in response. gpt-oss output varies between runs, so individual answers can change.
- The baseline scores 0 on conflict handling by construction, because it has no mechanism. Its stale-answer rate (0.20) is the meaningful comparison.
- **Code changed after this run** (not re-evaluated, because the daily quota was used up):
  - The verifier now receives each passage's document title, so "This procedure depends on …" supports a claim naming the procedure.
  - Re-attributing a claim to a passage the LLM did *not* cite now needs a support score of at least 0.8 (`CITATION_REPAIR_THRESHOLD`).

  Both fix a real bug found while taking screenshot 4. A correct citation was rejected, and a borderline NLI score (0.51) then moved it to an unrelated passage. Expect small differences in claim-support numbers on a re-run.
- An earlier mock-LLM run (extractive answers, so claim support is trivially 1.0 for both configurations) is superseded by this run.

### Threshold calibration

`scripts/calibrate_thresholds.py` collects each question's initial and post-recovery evidence (56 samples). It labels a sample *sufficient* when the question is answerable and a labelled relevant document is present. It then grid-searches `sufficiency_threshold × min_chunk_relevance × min_concept_coverage` by F1, breaking ties toward the most conservative values (a wrongly "sufficient" verdict is worse than an abstention), and reports leave-one-out accuracy. The latest run gave F1 0.961, balanced accuracy 0.963 and LOO accuracy 0.929, with recommended values 0.75 / 0.5 / 0.5 (`data/evaluation/calibration.json`, loaded automatically unless the env vars are set). Two of the three misclassifications (Q10/Q11 initial) are label artifacts: the right document was in the top-6 but ranked near zero, which is exactly the case recovery repairs.

---

## Tests

```bash
pytest                                     # unit + API integration (deterministic, no network)
pytest -rs                                 # also shows skip reasons
NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_PASSWORD=<pw> pytest tests/integration/test_neo4j_integration.py
```

- **Unit tests** use the in-memory store, hashing embeddings, a lexical reranker and verifier, and a mock or scripted LLM. They make no network or paid-API calls.
- **Challenge 1** (`test_recovery.py`, `test_evidence_assessment.py`): initial success; missed document recovered via the graph; query expansion recovery; absent document; irrelevant results; retry limit (1, 3, disabled); insufficient after recovery leads to abstention; documents without an answer.
- **Challenge 2** (`test_contradictions.py`): supersession; insufficient precedence metadata; compatible conditions; missing or incorrect metadata (circular supersession, unknown status or authority, not-yet-effective documents); conflicts across graph-connected documents; incorrectly extracted claims; disclosure of unresolved conflicts; configurable precedence.
- **Challenge 3** (`test_verification.py`): fully supported answer; one unsupported claim (corrected, or removed); invalid citation ID; citation to an unrelated passage (repaired); contradicted claim; unanswerable question (abstains with no LLM call); failed correction leads to abstention; baseline shows no verification; citation validation against the store.
- **Integration tests:**
  - `test_api.py`: the full HTTP flows, including upload, sanitisation, size limits, delete-then-query, all three challenges, and evaluation
  - `test_models_integration.py`: the real local models, including an end-to-end recovery on the full corpus
  - `test_neo4j_integration.py`: real Neo4j, covering constraints, the vector index, labels and relationship types, idempotent re-ingestion, traversal, the three challenges, and deletion cleanup

  **The Neo4j tests wipe the target database**, which is why they are opt-in.

Last run (2026-10-24): **112 passed, 0 failed, 0 skipped**, with `NEO4J_TEST_URI` set against Neo4j 5.26 and the models available. Without `NEO4J_TEST_URI`: 106 passed, 6 skipped (the opt-in Neo4j tests). The Groq client is unit-tested with a fake SDK (`test_llm_clients.py`: strict schema conversion, JSON-mode fallback, invalid-output re-ask, provider selection), so no test calls a paid API.

---

## Project structure

```
enterprise-graphrag-assistant/
├── app/
│   ├── main.py · config.py
│   ├── api/            routes_documents.py · routes_chat.py · routes_evaluation.py · dependencies.py
│   ├── ingestion/      parsers.py · metadata.py · chunking.py · pipeline.py
│   ├── graph/          store.py · neo4j_client.py · memory_store.py · schema.py · graph_queries.py
│   │                   entity_extraction.py · relationship_extraction.py
│   ├── retrieval/      vector_search.py · keyword_search.py · fusion.py · reranking.py · hybrid.py
│   │                   graph_retrieval.py · evidence_assessment.py · recovery.py
│   ├── reliability/    claim_extraction.py · contradiction_detection.py · conflict_resolution.py
│   │                   conflict_analysis.py · answer_verification.py · citations.py · abstention.py · nli.py
│   ├── generation/     llm_client.py · prompts.py · answer_generator.py
│   ├── evaluation/     datasets.py · custom_metrics.py · ragas_evaluation.py · runner.py · reporting.py
│   ├── models/         schemas.py
│   ├── services/       container.py · document_service.py · chat_service.py
│   └── utils/          text.py
├── frontend/streamlit_app.py
├── data/
│   ├── ontology.json              entity catalog, synonyms, claim property lexicon
│   ├── synthetic_documents/       13 documents (11 TXT, 1 PDF, 1 DOCX)
│   ├── synthetic_sources/         sources for the generated PDF/DOCX
│   └── evaluation/                eval_dataset.json · gold_conflicts.json · calibration.json · results/
├── scripts/  initialize_database.py · seed_synthetic_data.py · run_evaluation.py
│             calibrate_thresholds.py · generate_binary_documents.py
├── tests/    unit/ · integration/ · conftest.py
├── Dockerfile · docker-compose.yml · requirements.txt · pyproject.toml · .env.example
```

### Synthetic corpus

| ID | Document | Demonstrates |
|---|---|---|
| HR-POL-007-V1 / HR-POL-007 | Remote Work Policy v1.0 (superseded) / v2.0 | supersession: 2 vs 3 days/week, personal devices permitted vs prohibited |
| SEC-POL-001 | Information Security Policy | governs production systems; multi-hop target |
| SEC-STD-004 | Production Database Access Standard | recovery target for "Atlas" questions |
| TECH-DB-002 (PDF, 2 pages) | Database Platform Overview | "Atlas is a production database" (RELATED_TO edge); page numbers |
| TECH-API-002 | API Authentication Standard | DEPENDS_ON target |
| PROC-PART-003 (DOCX) | Partner Integration Onboarding Procedure | "which procedure depends on…" |
| PROC-SEC-002 | Security Incident Response Procedure | SUPPORTS SEC-POL-001 (1-hour reporting) |
| LEG-POL-003 vs SRE-STD-010 | Data Retention Policy vs Logging Standard | 12 months vs 90 days, resolved by authority |
| IT-STD-015 vs SEC-GDL-009 | Remote Access Standard vs Network Security Guidelines | 60 vs 15 min VPN timeout, **unresolved** (equal authority, missing effective date) |
| IT-STD-015 vs HR-POL-007 | webmail-only exception vs general ban | compatible conditions (not a contradiction) |
| ORG-REG-001 | Department Ownership Register | OWNED_BY edges |
| (none) | maternity leave, expenses, dress code, bonuses | absent topics, so the system should abstain |

---

## Design decisions

- **Custom orchestration instead of LangChain or LangGraph.** Each reliability step is a small, testable Python function, so the control flow (retry bounds, abstention) is explicit. Ragas pulls in `langchain-core` as its own dependency, but the application doesn't use it.
- **A storage interface with two implementations.** `Neo4jGraphStore` is the real backend. `InMemoryGraphStore` has the same semantics, including department-hub avoidance and provenance retraction, so unit tests are fast and deterministic. The Neo4j integration suite checks that the two behave the same.
- **BM25 in memory, rebuilt on change.** This is simple and always consistent with the graph for a small corpus. A Neo4j full-text index is also created as the scale-out path.
- **Rule-based extraction by default.** It is transparent, deterministic and testable. LLM extraction is optional and provenance-checked.
- **Citations built by the application, never by the LLM.** This makes invented document IDs or pages structurally impossible, not just discouraged by the prompt.
- **Conservative defaults.** Ties in calibration favour abstention, unresolved conflicts are disclosed, and newer documents don't automatically win.

## Limitations

- **Entity linking is catalog-based** (`ontology.json` plus document front matter). Entities outside the catalog aren't linked, so uploaded documents in new domains get weaker graph structure until the catalog is extended or LLM extraction is enabled.
- **Claim extraction is heuristic.** It needs explicit modal verbs. Subjects fall back to the document's own policy entity. Properties come from a keyword lexicon, so a wrong property can hide a conflict or (rarely) pair unrelated rules. Unstructured claims are only compared when an NLI model is loaded, and small NLI models miss subtle contradictions and struggle with numbers. That is why explicit number checks exist.
- **Concept coverage is lexical with a synonym list.** Paraphrases outside the list rely on the cross-encoder waiver. Expect false abstentions on unusual phrasing.
- **Condition handling is coarse.** Any difference in condition text makes rules "compatible". Overlapping-but-different conditions aren't reasoned about.
- **Verification is only as good as the verifier.** NLI over sentence windows can accept a claim that combines two passages incorrectly, and the lexical fallback can't detect paraphrase.
- **The mock LLM does no reasoning.** It exists for offline tests and demos only. Groq generation, correction and Ragas were exercised. The Gemini provider, `VERIFIER_BACKEND=llm` and `EXTRACTION_BACKEND=llm` are implemented but were not exercised.
- **LLM output varies between runs.** gpt-oss output differs from run to run even at temperature 0, so individual answers (and borderline metrics) can change. Answers that fuse two passages into one claim (for example, applying a production-database rule "to Atlas") are rejected by the verifier; the bounded correction usually fixes them, and otherwise the system abstains.
- **Not production-hardened.** There is no authentication or authorisation on the API, the in-memory BM25 is single-process, evaluation runs synchronously, and there is no rate limiting.
- **Dependency notes.** ragas 0.4.3 needs `langchain-community<0.4` and a recent `instructor` (1.17), with `openai<3` for `langchain-openai`; the pins are in `requirements.txt`. `ragas.evaluate()` is deprecated in 0.4, so the per-sample `ascore` collections API is used instead.
