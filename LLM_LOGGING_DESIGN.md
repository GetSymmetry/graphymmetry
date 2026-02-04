# LLM Call Logging Design

## Goal
Enable per-episode LLM call logging to profile what happens during episode processing.

## Use Case
```python
# Profile a single episode
result = await client.add_episode(
    name="turn_5",
    episode_body="...",
    enable_llm_logging=True,
    llm_log_path="logs/llm_calls_20260203_103045_turn_5.jsonl"
)
```

Output: `logs/llm_calls_20260203_103045_turn_5.jsonl`
```jsonl
{"timestamp": "2026-02-03T10:30:45.123Z", "model": "gpt-4o", "method": "POST", "url": "/chat/completions", "status": 200, "duration_ms": 234}
{"timestamp": "2026-02-03T10:30:46.456Z", "model": "gpt-4o-mini", "method": "POST", "url": "/chat/completions", "status": 200, "duration_ms": 145}
```

## Architecture

### 1. Add logging parameters to `add_episode()`

**File**: `graphiti_core/graphiti.py`

```python
async def add_episode(
    self,
    name: str,
    episode_body: str,
    ...,
    enable_llm_logging: bool = False,
    llm_log_path: str | None = None,
):
    """
    Args:
        enable_llm_logging: Enable detailed logging of all LLM calls for this episode
        llm_log_path: Path to JSONL file for logging (required if enable_llm_logging=True)
    """
```

### 2. Create logging context manager

**File**: `graphiti_core/llm_client/logging.py` (new file)

```python
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

class LLMCallLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_file = None

    async def log_request(self, request: httpx.Request):
        """Log before request is sent"""
        request.extensions['start_time'] = time.time()

    async def log_response(self, response: httpx.Response):
        """Log after response is received"""
        duration_ms = (time.time() - response.request.extensions['start_time']) * 1000

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": response.request.method,
            "url": str(response.request.url),
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
            # Extract model from request body if available
            "model": self._extract_model(response.request),
        }

        # Append to JSONL file
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()

    def _extract_model(self, request: httpx.Request) -> str | None:
        """Extract model name from request body"""
        try:
            if request.content:
                body = json.loads(request.content)
                return body.get("model")
        except:
            pass
        return None

    @asynccontextmanager
    async def enable(self, client: httpx.AsyncClient):
        """Enable logging for httpx client"""
        # Open log file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'a')

        # Add event hooks
        client.event_hooks['request'].append(self.log_request)
        client.event_hooks['response'].append(self.log_response)

        try:
            yield
        finally:
            # Remove hooks
            client.event_hooks['request'].remove(self.log_request)
            client.event_hooks['response'].remove(self.log_response)

            # Close file
            if self.log_file:
                self.log_file.close()
```

### 3. Enable logging in `add_episode()`

**File**: `graphiti_core/graphiti.py`

```python
async def add_episode(
    self,
    ...
    enable_llm_logging: bool = False,
    llm_log_path: str | None = None,
):
    if enable_llm_logging:
        if not llm_log_path:
            raise ValueError("llm_log_path required when enable_llm_logging=True")

        # Get httpx client from AsyncOpenAI
        httpx_client = self.llm_client.client._client  # Access internal httpx client

        # Enable logging for this episode
        logger = LLMCallLogger(llm_log_path)
        async with logger.enable(httpx_client):
            # All LLM calls within this context will be logged
            result = await self._process_episode(...)
    else:
        result = await self._process_episode(...)

    return result
```

## Integration with graphiti_eval

**File**: `graphiti_eval/src/ingestion/turn_by_turn/processor.py`

```python
async def ingest_turn(
    client: GraphitiClient,
    turn: Turn,
    group_id: str = "evaluation",
    saga_id: Optional[str] = None,
    previous_episode_uuids: Optional[list[str]] = None,
    enable_llm_logging: bool = False,
) -> IngestionResult:
    """
    Args:
        enable_llm_logging: Enable detailed LLM call logging for profiling
    """

    # Generate log path if logging enabled
    llm_log_path = None
    if enable_llm_logging:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        llm_log_path = f"logs/llm_calls_{timestamp}_{turn.turn_id}.jsonl"

    result = await client.client.add_episode(
        ...,
        enable_llm_logging=enable_llm_logging,
        llm_log_path=llm_log_path,
    )
```

Then in `parallel_ingestion_processor.py`:
```bash
python scripts/parallel_ingestion_processor.py \
  --profile-llm-calls \
  --max-conversations 1  # Profile just one conversation
```

## Log File Format

Each line is a JSON object:
```json
{
  "timestamp": "2026-02-03T10:30:45.123456Z",
  "method": "POST",
  "url": "https://api.openai.com/v1/chat/completions",
  "status": 200,
  "duration_ms": 234.56,
  "model": "gpt-4o"
}
```

## File Naming Convention

```
logs/llm_calls_{timestamp}_{episode_name}.jsonl
```

Examples:
- `logs/llm_calls_20260203_103045_turn_5.jsonl`
- `logs/llm_calls_20260203_103100_chunk_001.jsonl`

## Future: Profile Analyzer

Later, create a separate script to analyze the logs:
```bash
python scripts/analyze_llm_profile.py logs/llm_calls_20260203_103045_turn_5.jsonl
```

Output:
- Timeline visualization
- Call frequency over time
- Model usage breakdown
- Latency analysis
- Request bursts/patterns

## Open Questions

1. **Should logging be always-on or opt-in?**
   - Proposal: Opt-in per episode (as shown above)

2. **Should we log request/response bodies?**
   - Proposal: No - too verbose, privacy concerns
   - Just log metadata (model, timing, status)

3. **What about embedding calls?**
   - Same approach - httpx hooks will catch all HTTP calls
   - Can distinguish by URL (/chat/completions vs /embeddings)

4. **Thread safety for parallel episodes?**
   - Each episode gets its own log file (named by episode)
   - No contention issues
