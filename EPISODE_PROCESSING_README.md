# Episode Processing Deep Dive

This document explains the complete processing pipeline that occurs when you call `add_episode()` in Graphiti. Understanding this flow is crucial for optimizing ingestion performance and debugging extraction issues.

## Table of Contents

- [Overview](#overview)
- [Stage 1: Context Retrieval](#stage-1-context-retrieval)
- [Stage 2: Entity Extraction with Adaptive Chunking](#stage-2-entity-extraction-with-adaptive-chunking)
- [Stage 3: Entity Merging and Deduplication](#stage-3-entity-merging-and-deduplication)
- [Stage 4: Edge Extraction with Covering Chunks](#stage-4-edge-extraction-with-covering-chunks)
- [Stage 5: Attribute Extraction and Graph Update](#stage-5-attribute-extraction-and-graph-update)
- [Configuration Parameters](#configuration-parameters)
- [Performance Considerations](#performance-considerations)

---

## Overview

When you call `add_episode()`, Graphiti performs a sophisticated multi-stage pipeline to extract structured knowledge from unstructured content. The process is designed to handle episodes of any size while maintaining quality and avoiding LLM context limits.

```
Episode Content
    ↓
[1] Retrieve Previous Episodes (context)
    ↓
[2] Extract Entities (with adaptive chunking)
    ↓
[3] Merge & Deduplicate Entities (against existing graph)
    ↓
[4] Extract Edges (with covering chunks)
    ↓
[5] Extract Attributes & Update Graph
    ↓
Graph Updated
```

---

## Stage 1: Context Retrieval

**Location**: `graphiti.py:951-961`

Before processing the new episode, Graphiti retrieves **previous episodes** for context.

### What Gets Retrieved

```python
previous_episodes = await self.retrieve_episodes(
    reference_time,
    last_n=RELEVANT_SCHEMA_LIMIT,  # Default: 10
    group_ids=[group_id],
    source=source,
)
```

- **Count**: Last 10 episodes (configurable via `RELEVANT_SCHEMA_LIMIT`)
- **Ordering**: By timestamp relative to `reference_time`
- **Content**: Full episode content, not just metadata

### Purpose

Previous episodes provide **conversational context** to the extraction LLMs:
- Recent entities and topics discussed
- Continuation of narrative threads
- Temporal context for entity references

**Important**: This is **NOT** used for deduplication—that happens separately via graph search (see Stage 3).

---

## Stage 2: Entity Extraction with Adaptive Chunking

**Location**: `utils/maintenance/node_operations.py:66-113`

Entity extraction uses **adaptive chunking** to handle large episodes without hitting LLM context limits or causing timeouts.

### Decision: To Chunk or Not to Chunk

**Location**: `utils/content_chunking.py:59-83`

```python
if should_chunk(episode.content, episode.source):
    extracted_entities = await _extract_nodes_chunked(llm_client, episode, context)
else:
    extracted_entities = await _extract_nodes_single(llm_client, episode, context)
```

Chunking is triggered when **BOTH** conditions are met:

1. **Size threshold**: Episode >= 1000 tokens (default `CHUNK_MIN_TOKENS`)
2. **High entity density**:
   - **JSON**: High ratio of array elements or object keys per token
   - **Text**: High ratio of capitalized words (proxy for named entities)
   - **Threshold**: 0.15 density (configurable via `CHUNK_DENSITY_THRESHOLD`)

### Why Density Matters

- **High-density content** (data dumps, entity lists): Benefits from chunking, avoids LLM timeouts
- **Low-density content** (prose, narratives): Processes fine as-is, maintains context

### Chunking Strategy

**Location**: `utils/content_chunking.py:376-450` (text), `215-251` (JSON), `549-593` (messages)

Different content types use different chunking strategies:

#### Text Content
- Splits at natural boundaries: paragraphs → sentences → words
- **Chunk size**: 3000 tokens (default `CHUNK_TOKEN_SIZE`)
- **Overlap**: 200 tokens between chunks (default `CHUNK_OVERLAP_TOKENS`)
- Overlap captures entities at boundaries

#### JSON Content
- Splits at element/key boundaries (never mid-object)
- Arrays: grouped by elements
- Objects: grouped by top-level keys

#### Message Content
- Never splits mid-message
- Preserves speaker boundaries
- Handles formats: JSON arrays, "Speaker: text", line-separated

### Parallel Extraction

**Location**: `utils/maintenance/node_operations.py:155-182`

When chunked, extraction happens **in parallel**:

```python
chunk_results = await semaphore_gather(
    *[_extract_from_chunk(llm_client, chunk, context, episode) for chunk in chunks]
)
```

Each chunk receives:
- The **chunk content** (subset of episode)
- **Full previous episodes context** (all 10)
- **Custom extraction instructions** (if provided)
- **Entity type definitions**

### Merging Chunk Results

**Location**: `utils/maintenance/node_operations.py:225-242`

After parallel extraction, entities are merged with **simple case-insensitive deduplication**:

```python
def _merge_extracted_entities(chunk_results):
    seen_names = set()
    merged = []

    for entities in chunk_results:
        for entity in entities:
            normalized = entity.name.strip().lower()
            if normalized and normalized not in seen_names:
                seen_names.add(normalized)
                merged.append(entity)  # Keep first occurrence

    return merged
```

**Key points**:
- Case-insensitive: "Apple Inc" == "apple inc"
- Keeps first occurrence (chunk order matters)
- Fast, no LLM calls at this stage

---

## Stage 3: Entity Merging and Deduplication

**Location**: `utils/maintenance/node_operations.py:465-515`

After initial extraction, entities are resolved against the **existing graph** to avoid duplicates.

### Step 3.1: Search for Candidates

**Location**: `utils/maintenance/node_operations.py:475-479`

For each extracted entity, Graphiti searches the graph:

```python
search_results = await search(
    query=node.name,  # e.g., "Apple Inc"
    group_ids=[node.group_id],
    config=NODE_HYBRID_SEARCH_RRF,  # Vector + keyword search
)
candidate_nodes = [node for result in search_results for node in result.nodes]
```

**Search scope**: Entire graph within the group (not just recent episodes)
**Search type**: Hybrid (combines vector embeddings + keyword matching)

### Step 3.2: Deterministic Similarity Resolution

**Location**: `utils/maintenance/node_operations.py:489`

Uses **embedding cosine similarity** to resolve clear matches:

```python
_resolve_with_similarity(extracted_nodes, indexes, state)
```

If an existing entity has high similarity (above threshold), the extracted entity is merged with it. This handles variations like:
- "Dr. Smith" vs "Dr Smith"
- "OpenAI" vs "OpenAI Inc."
- "Alice Johnson" vs "Alice M. Johnson"

### Step 3.3: LLM-Based Deduplication

**Location**: `utils/maintenance/node_operations.py:491-499`

For **ambiguous cases**, an LLM decides:

```python
await _resolve_with_llm(
    llm_client,
    extracted_nodes,
    indexes,
    state,
    episode,
    previous_episodes,
    entity_types,
)
```

The LLM receives:
- Current episode content
- Previous episodes context
- Extracted entity
- Candidate existing entities

It decides: "Is extracted entity X the same as existing entity Y?"

### UUID Mapping

When a match is found:

```python
state.uuid_map[new_entity_uuid] = existing_entity_uuid
```

This mapping ensures edges reference the correct (deduplicated) entity UUIDs.

---

## Stage 4: Edge Extraction with Covering Chunks

**Location**: `utils/maintenance/edge_operations.py:92-231`

Edge extraction uses a **different chunking strategy** than entity extraction—it's based on **entity pairs**, not content size.

### The Pair Explosion Problem

With N entities, there are `N × (N-1) / 2` possible pairs:
- 5 entities → 10 pairs
- 50 entities → 1,225 pairs
- 100 entities → 4,950 pairs

You can't fit all pairs in one LLM prompt!

### Covering Chunks Solution

**Location**: `utils/content_chunking.py:719-826`

Graphiti generates **covering chunks** that ensure every pair appears in at least one chunk:

```python
covering_chunks = generate_covering_chunks(nodes, MAX_NODES=10)
```

**Algorithm**: Greedy approach based on the Handshake Flights Problem
- Each chunk contains up to 10 entities
- Chunks may overlap (entities appear in multiple chunks)
- Every pair is assigned to exactly one chunk (no duplicate edge extraction)

**Example**:
```
Entities: [Alice, Bob, Carol, Dave, Eve, Frank]  (15 pairs total)

Chunk 1: [Alice, Bob, Carol, Dave]
  Pairs: (Alice,Bob), (Alice,Carol), (Alice,Dave),
         (Bob,Carol), (Bob,Dave), (Carol,Dave)  [6 pairs]

Chunk 2: [Alice, Eve, Frank, Carol]
  Pairs: (Alice,Eve), (Alice,Frank), (Eve,Frank),
         (Eve,Carol), (Frank,Carol)  [5 pairs]

Chunk 3: [Bob, Eve, Frank, Dave]
  Pairs: (Bob,Eve), (Bob,Frank), (Dave,Eve),
         (Dave,Frank)  [4 pairs]

Total: 15 pairs covered
```

### Edge Extraction Per Chunk

**Location**: `utils/maintenance/edge_operations.py:145-231`

Each covering chunk is processed:

```python
context = {
    'episode_content': episode.content,  # FULL episode, not chunked
    'nodes': [{'id': idx, 'name': node.name} for idx, node in enumerate(chunk)],
    'previous_episodes': [ep.content for ep in previous_episodes],
    'edge_types': edge_types_context,
}
```

**Key point**: The LLM sees the **full episode content**, not a chunk. It's only the list of entities that's chunked.

### LLM Task

Given the full episode and a list of entities, the LLM:
1. Examines every pair in the chunk
2. Finds relationships mentioned/implied in the episode
3. Returns only pairs with actual relationships

**Example output**:
```json
{
  "edges": [
    {
      "source_entity_id": 0,
      "target_entity_id": 1,
      "relation_type": "WORKS_WITH",
      "fact": "Alice works with Bob on the project"
    },
    {
      "source_entity_id": 2,
      "target_entity_id": 3,
      "relation_type": "REPORTS_TO",
      "fact": "Carol reports to Dave"
    }
  ]
}
```

Not every pair needs a relationship—only those actually present in the content.

### No Duplicate Edges

Since pairs are uniquely assigned to chunks, there's no need to deduplicate edges. Each pair is examined exactly once.

---

## Stage 5: Attribute Extraction and Graph Update

**Location**: `utils/maintenance/node_operations.py:533-565`

After entity and edge extraction, Graphiti:

1. **Extracts attributes** for each entity (summaries, metadata)
2. **Creates embeddings** for entities
3. **Saves to graph database**:
   - Episode node
   - Entity nodes (deduplicated)
   - Edge relationships
   - Episodic edges (HAS_MENTION, etc.)
4. **Updates communities** (if requested)

---

## Configuration Parameters

All parameters can be set via environment variables:

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_TOKEN_SIZE` | 3000 | Target size per chunk in tokens |
| `CHUNK_OVERLAP_TOKENS` | 200 | Overlap between chunks (captures boundary entities) |
| `CHUNK_MIN_TOKENS` | 1000 | Minimum tokens before considering chunking |
| `CHUNK_DENSITY_THRESHOLD` | 0.15 | Entity density threshold (chunk if exceeded) |

**Location**: `graphiti_core/helpers.py:43-53`

### Context Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RELEVANT_SCHEMA_LIMIT` | 10 | Number of previous episodes to retrieve for context |

**Location**: `graphiti_core/search/search_utils.py:63`

### Edge Extraction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_NODES` | 10 | Maximum entities per covering chunk |

**Location**: `utils/maintenance/edge_operations.py` (hardcoded constant)

---

## Performance Considerations

### When Chunking Helps

✅ **Large data dumps** (JSON with many objects)
✅ **Entity-dense content** (lists, directories, structured data)
✅ **Very long documents** (>3000 tokens)

### When Chunking Hurts

❌ **Short content** (<1000 tokens) - overhead not worth it
❌ **Narrative prose** - loses context across chunks
❌ **Content with few entities** - low density, processes fine as-is

### Optimization Tips

1. **Preprocess before ingestion**: If you're already chunking semantically (like conversation preprocessing), your chunks are likely already optimal. Graphiti will only sub-chunk if individual chunks are very large.

2. **Adjust chunk size**: If you have extremely entity-dense content causing timeouts, reduce `CHUNK_TOKEN_SIZE`.

3. **Parallel episodes**: Process multiple episodes concurrently using `add_episodes_batch()` or `add_episode_bulk()` to maximize throughput.

4. **Monitor metrics**: Use the returned `EpisodeMetrics` to track:
   - Entity extraction time
   - Edge extraction time
   - LLM call counts
   - Token usage

### Parallel Processing Safety

**Entity extraction chunks**: Parallel processing is safe because:
- Each chunk gets full previous episode context
- Overlap captures boundary entities
- Simple name deduplication merges results

**Edge covering chunks**: Parallel processing is safe because:
- Each chunk gets full episode content
- Pairs are uniquely assigned (no duplicates)
- Each LLM call has complete context

---

## Example: Complete Flow

```python
# Your code
await client.add_episode(
    name="meeting_notes",
    episode_body=large_document,  # 5000 tokens, high entity density
    reference_time=datetime.now(),
    group_id="my_project",
)
```

**What happens internally**:

1. **Context**: Retrieve last 10 episodes from "my_project"

2. **Entity Extraction**:
   - `should_chunk()` → True (5000 tokens, high density)
   - Split into 2 chunks: [0-3200], [3000-5000] (200 token overlap)
   - Extract entities from each chunk in parallel
   - Chunk 1: [Alice, Bob, Corp, Product A, Feature X, ...]
   - Chunk 2: [Product A, Feature X, Feature Y, Team, Carol, ...]
   - Merge: Dedupe by name → [Alice, Bob, Corp, Product A, Feature X, Feature Y, Team, Carol]

3. **Entity Deduplication**:
   - Search graph for each: "Alice", "Bob", "Corp", etc.
   - Find: "Alice" exists (UUID: abc-123)
   - Find: "Corp" exists as "Corporation" (UUID: def-456) via LLM dedupe
   - Create: "Feature X", "Feature Y", "Team", "Carol" (new)
   - UUID map: {new_alice_uuid → abc-123, new_corp_uuid → def-456}

4. **Edge Extraction**:
   - 8 entities → 28 pairs
   - Generate 4 covering chunks of ~10 entities each
   - Each chunk gets full 5000-token episode
   - Extract edges in parallel
   - Find: (Alice, Corp, WORKS_FOR), (Bob, Team, MEMBER_OF), ...

5. **Save**:
   - Create episode node
   - Save/update 8 entities (4 new, 4 merged)
   - Save 12 edges
   - Create embeddings

**Result**: Structured knowledge graph updated with deduplicated entities and relationships.

---

## Debugging Tips

### Low Entity Extraction

**Check**: Is density too low for chunking to trigger?
- Set `CHUNK_MIN_TOKENS=0` to force chunking
- Review `should_chunk()` logic for your content type

### Duplicate Entities

**Check**: Is deduplication working?
- Verify entities are in the same `group_id`
- Check embedding similarity threshold
- Review LLM dedupe prompts

### Missing Edges

**Check**: Are entities too far apart?
- Review covering chunks (are relevant pairs in the same chunk?)
- Check if episode content mentions the relationship
- Increase `MAX_NODES` to include more entities per chunk

### Timeouts

**Check**: Are chunks too large?
- Reduce `CHUNK_TOKEN_SIZE`
- Check entity density (`CHUNK_DENSITY_THRESHOLD`)
- Monitor LLM token usage in metrics

---

## Related Documentation

- [Main README](README.md) - Getting started with Graphiti
- [OTEL_TRACING.md](OTEL_TRACING.md) - Observability and tracing
- [AGENTS.md](AGENTS.md) - Building agent systems with Graphiti

---

**Questions or issues?** Open an issue on GitHub or join the community discussions.
