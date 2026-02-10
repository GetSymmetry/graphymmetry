# Graphymmetry Integration Guide

Swapping from `graphiti-core` (upstream getzep/graphiti) to `graphymmetry` (GetSymmetry fork).

## Compatibility

Graphymmetry is a **pure superset** of graphiti-core at fork point `d9aae86`. No APIs were changed or removed. Existing code works as-is — the additions below are all opt-in.

## Step 1: Swap the Dependency

In `pyproject.toml`, change the source for `graphiti-core`:

```toml
[project]
dependencies = [
    "graphiti-core",
    # ... other deps
]

[tool.uv.sources]
graphiti-core = { path = "../graphymmetry", editable = true }
```

That's it for basic compatibility. Everything below is about adopting the new features.

---

## Step 2: Adopt Rate Limiting

**Why:** Prevents hitting API rate limits during parallel ingestion.

```python
from graphiti_core import Graphiti, RateLimitConfig

rate_limit_config = RateLimitConfig(
    max_concurrent_medium=40,    # Large LLM calls (gpt-4, claude-opus)
    max_concurrent_small=40,     # Small LLM calls (gpt-4-mini, haiku)
    max_concurrent_embeddings=250,  # Embedding API calls
)

client = Graphiti(
    uri=neo4j_uri,
    user=neo4j_user,
    password=neo4j_password,
    llm_client=llm_client,
    embedder=embedder,
    rate_limit_config=rate_limit_config,  # NEW
)
```

The rate limiter is shared across the LLM client and embedder automatically. No other code changes needed.

**Reference:** `graphiti_eval/src/client.py`, `graphiti_eval/scripts/ingest.py`

---

## Step 3: Adopt LLM Call Logging

**Why:** Produces per-episode JSONL traces of every LLM and embedding call — useful for debugging extraction quality and profiling costs.

### On `add_episode()`

```python
result = await client.add_episode(
    name="turn_001",
    episode_body=content,
    source_description="conversation",
    reference_time=timestamp,
    # ... existing params ...
    enable_llm_logging=True,                          # NEW
    llm_log_path="logs/llm_calls_turn_001.jsonl",     # NEW
)
```

### On `add_episode_bulk()`

```python
result = await client.add_episode_bulk(
    bulk_episodes=episodes,
    group_id=group_id,
    # ... existing params ...
    enable_llm_logging=True,                       # NEW
    llm_log_path="logs/llm_calls_bulk.jsonl",      # NEW
)
```

`enable_llm_logging` defaults to `False`. When enabled, `llm_log_path` is required.

Each line in the JSONL file contains: timestamp, model, duration, status, token usage, and request/response metadata.

**Reference:** `graphiti_eval/src/ingestion/turn_by_turn/processor.py`, `graphiti_eval/src/ingestion/preprocessor/ingest.py`

---

## Step 4: Use Stagger Delay for Bulk Ingestion

**Why:** Spreads out extraction requests in `add_episode_bulk()` to avoid API rate limit bursts.

```python
result = await client.add_episode_bulk(
    bulk_episodes=episodes,
    group_id=group_id,
    # ... existing params ...
    stagger_delay=6.0,  # NEW — seconds between episode extractions
)
```

Only relevant when using `add_episode_bulk()`. The rate limiter (Step 2) handles concurrency at the semaphore level; stagger delay adds coarse-grained spacing on top.

**Reference:** `graphiti_eval/src/ingestion/preprocessor/ingest.py:272`

---

## Available but Not Yet Adopted

These features exist in graphymmetry but `graphiti_eval` doesn't use them yet. Consider them for future work.

### Batch Processing (`add_episodes_batch`)

Parallel episode processing with callbacks and error control. Different from `add_episode_bulk()` — each episode is processed independently (no shared dedup across the batch).

```python
from graphiti_core import EpisodeInput, BatchResults, ErrorAction

episodes = [
    EpisodeInput(
        name="turn_001",
        episode_body=content,
        source_description="conversation",
        reference_time=timestamp,
    ),
    # ...
]

results: BatchResults = await client.add_episodes_batch(
    episodes=episodes,
    parallelism=5,
    on_episode_complete=lambda name, metrics: print(f"{name}: {metrics.wall_clock_ms}ms"),
    on_error=lambda name, exc: ErrorAction.SKIP_EPISODE,
)

print(f"Succeeded: {len(results.successful)}, Failed: {len(results.failed)}")
```

### Metrics Collection

Per-episode telemetry is returned in `AddEpisodeResults.metrics` (an `EpisodeMetrics` instance) when using `add_episode()`. Fields include:

- `wall_clock_ms` — total processing time
- `llm_calls_medium` / `llm_calls_small` — LLM call counts by model size
- `embedding_calls` — embedding API calls
- `entities_extracted` / `edges_extracted` — extraction counts
- `retries` / `errors` — error tracking

`BatchMetrics` aggregates these across a batch when using `add_episodes_batch()`.

---

## Quick Reference: New Imports

```python
# Rate limiting
from graphiti_core import RateLimitConfig

# Batch processing (not yet used in graphiti_eval)
from graphiti_core import EpisodeInput, BatchResults, ErrorAction

# Metrics (not yet used in graphiti_eval)
from graphiti_core import EpisodeMetrics, BatchMetrics, MetricsCollector

# LLM logging (used internally, rarely imported directly)
from graphiti_core.llm_client.logging import LLMCallLogger
```

## Other Internal Changes

These require no code changes but are good to know about:

- `MAX_NODES` in edge extraction increased from 15 to 40 (more entities processed per chunk)
- `MAX_RETRIES` on Azure OpenAI client increased to 4
- Retry logic added for rate limit errors across all LLM clients
- Rate limiting integrated into embedder base class (`create()` / `create_batch()`)
- Context-var based logger isolation for safe concurrent async usage
