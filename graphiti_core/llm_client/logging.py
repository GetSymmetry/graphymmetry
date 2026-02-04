"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMCallLogger:
    """Logger for LLM API calls at the HTTP level.

    Uses httpx event hooks to capture request/response metadata for profiling.
    Logs to JSONL format (one JSON object per line).
    """

    def __init__(self, log_path: str):
        """Initialize logger with output path.

        Args:
            log_path: Path to JSONL file for logging
        """
        self.log_path = Path(log_path)
        self.log_file = None

    async def _log_request(self, request: httpx.Request) -> None:
        """Hook called before request is sent.

        Stores start time in request extensions for duration calculation.
        """
        request.extensions['llm_log_start_time'] = time.time()

    async def _log_response(self, response: httpx.Response) -> None:
        """Hook called after response is received.

        Logs request/response metadata to JSONL file.
        """
        start_time = response.request.extensions.get('llm_log_start_time')
        if start_time is None:
            # Request wasn't tracked (shouldn't happen)
            return

        duration_ms = (time.time() - start_time) * 1000

        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'method': response.request.method,
            'url': str(response.request.url.path),  # Just path, not full URL with credentials
            'status': response.status_code,
            'duration_ms': round(duration_ms, 2),
            'model': self._extract_model(response.request),
        }

        # Append to JSONL file
        if self.log_file:
            self.log_file.write(json.dumps(log_entry) + '\n')
            self.log_file.flush()

    def _extract_model(self, request: httpx.Request) -> str | None:
        """Extract model name from request body.

        Args:
            request: httpx Request object

        Returns:
            Model name if found, None otherwise
        """
        try:
            if request.content:
                body = json.loads(request.content)
                return body.get('model')
        except Exception:
            pass
        return None

    @asynccontextmanager
    async def enable(self, client: httpx.AsyncClient):
        """Enable logging for httpx client within context.

        Args:
            client: httpx.AsyncClient to instrument

        Yields:
            None

        Example:
            >>> logger = LLMCallLogger("logs/llm_calls.jsonl")
            >>> async with logger.enable(httpx_client):
            ...     # All HTTP calls within this context are logged
            ...     await httpx_client.get("https://api.example.com")
        """
        # Create log directory if needed
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file in append mode
        self.log_file = open(self.log_path, 'a')

        # Add event hooks to client
        client.event_hooks['request'].append(self._log_request)
        client.event_hooks['response'].append(self._log_response)

        try:
            yield
        finally:
            # Remove hooks
            try:
                client.event_hooks['request'].remove(self._log_request)
                client.event_hooks['response'].remove(self._log_response)
            except ValueError:
                # Hooks already removed (shouldn't happen but handle gracefully)
                pass

            # Close file
            if self.log_file:
                self.log_file.close()
                self.log_file = None


async def get_httpx_client_from_openai(openai_client: Any) -> httpx.AsyncClient | None:
    """Extract httpx.AsyncClient from OpenAI client.

    OpenAI's AsyncOpenAI and AsyncAzureOpenAI clients wrap an httpx.AsyncClient
    internally. This helper extracts it for instrumentation.

    Args:
        openai_client: AsyncOpenAI or AsyncAzureOpenAI instance

    Returns:
        httpx.AsyncClient if found, None otherwise
    """
    try:
        # OpenAI client stores httpx client as ._client
        if hasattr(openai_client, '_client'):
            httpx_client = openai_client._client
            if isinstance(httpx_client, httpx.AsyncClient):
                return httpx_client
    except Exception as e:
        logger.warning(f'Failed to extract httpx client from OpenAI client: {e}')

    return None
