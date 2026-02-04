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
import threading
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

logger = logging.getLogger(__name__)

# Context variable for async-safe per-task call logger storage
# This allows each async task (e.g., parallel episode processing) to have
# its own isolated logger without race conditions on shared client instances
_call_logger_var: ContextVar['LLMCallLogger | None'] = ContextVar('call_logger', default=None)


def get_call_logger() -> 'LLMCallLogger | None':
    """Get the current task's call logger from context."""
    return _call_logger_var.get()


def set_call_logger(logger: 'LLMCallLogger | None') -> Token:
    """Set the call logger for the current task context.

    Args:
        logger: The LLMCallLogger to use, or None to disable call logging.

    Returns:
        A token that can be used with reset_call_logger() to restore the previous value.
    """
    return _call_logger_var.set(logger)


def reset_call_logger(token: Token) -> None:
    """Reset the call logger to its previous value using the token from set_call_logger().

    Args:
        token: The token returned from set_call_logger().
    """
    _call_logger_var.reset(token)


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
        self._lock = threading.Lock()

    def log_call(
        self,
        model: str,
        duration_ms: float,
        tokens_in: int,
        tokens_out: int,
        status: int = 200,
    ) -> None:
        """Log an LLM call with token usage.

        This method is called directly from LLM clients after each API call,
        providing accurate token counts from the response object.

        Thread-safe: uses internal lock for concurrent write protection.

        Args:
            model: Name of the model used
            duration_ms: Request duration in milliseconds
            tokens_in: Number of input/prompt tokens
            tokens_out: Number of output/completion tokens
            status: HTTP status code (default 200)
        """
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model': model,
            'duration_ms': round(duration_ms, 2),
            'status': status,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
        }
        if self.log_file:
            with self._lock:
                self.log_file.write(json.dumps(log_entry) + '\n')
                self.log_file.flush()

    def log_retry(
        self,
        model: str,
        attempt: int,
        max_retries: int,
        retry_delay: float,
        error_type: str = 'rate_limit',
    ) -> None:
        """Log a retry attempt before it happens.

        Thread-safe: uses internal lock for concurrent write protection.

        Args:
            model: Name of the model that triggered retry
            attempt: Current retry attempt number (1-indexed)
            max_retries: Maximum retries configured
            retry_delay: Delay in seconds before retry
            error_type: Type of error triggering retry ('rate_limit' or 'transient')
        """
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': 'retry',  # Distinguishes from regular calls
            'model': model,
            'error_type': error_type,
            'attempt': attempt,
            'max_retries': max_retries,
            'retry_delay_s': round(retry_delay, 2),
        }
        if self.log_file:
            with self._lock:
                self.log_file.write(json.dumps(log_entry) + '\n')
                self.log_file.flush()

    @asynccontextmanager
    async def open(self):
        """Open logger for direct logging without httpx hooks.

        Use this context manager when logging from LLM client level
        rather than via httpx event hooks.

        Example:
            >>> logger = LLMCallLogger("logs/llm_calls.jsonl")
            >>> async with logger.open():
            ...     logger.log_call(model="gpt-4", duration_ms=100, tokens_in=50, tokens_out=20)
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'a')
        try:
            yield self
        finally:
            if self.log_file:
                self.log_file.close()
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
