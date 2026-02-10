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

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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
    """Non-blocking logger for LLM API calls.

    All log entries are enqueued and processed by a background asyncio task,
    so logging never blocks the caller. Entries are routed to a sink function
    which is either a custom callable (for telemetry, Application Insights, etc.)
    or an internal file sink that writes JSONL.

    Provide exactly one of ``log_path`` or ``sink`` (mutually exclusive).

    Args:
        log_path: Path to JSONL file for logging. Creates an internal file sink.
        sink: Custom callable that receives log entry dicts. Must accept a single
            ``dict`` argument. Does not need to be thread-safe — it is only ever
            called from the background drain task.
    """

    def __init__(
        self,
        log_path: str | None = None,
        sink: Callable[[dict], None] | None = None,
    ):
        if log_path and sink:
            raise ValueError('Provide log_path or sink, not both.')
        if not log_path and not sink:
            raise ValueError('Provide either log_path or sink.')

        self._log_path = Path(log_path) if log_path else None
        self._log_file = None
        self._sink = sink or self._file_sink
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._drain_task: asyncio.Task | None = None

    def _file_sink(self, entry: dict) -> None:
        """Built-in sink that writes JSONL to the configured log file."""
        if self._log_file:
            self._log_file.write(json.dumps(entry) + '\n')
            self._log_file.flush()

    def log_call(
        self,
        model: str,
        duration_ms: float,
        tokens_in: int,
        tokens_out: int,
        status: int = 200,
    ) -> None:
        """Log an LLM call with token usage.

        Non-blocking: enqueues the entry for background processing.

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
        self._queue.put_nowait(log_entry)

    def log_retry(
        self,
        model: str,
        attempt: int,
        max_retries: int,
        retry_delay: float,
        error_type: str = 'rate_limit',
    ) -> None:
        """Log a retry attempt before it happens.

        Non-blocking: enqueues the entry for background processing.

        Args:
            model: Name of the model that triggered retry
            attempt: Current retry attempt number (1-indexed)
            max_retries: Maximum retries configured
            retry_delay: Delay in seconds before retry
            error_type: Type of error triggering retry ('rate_limit' or 'transient')
        """
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': 'retry',
            'model': model,
            'error_type': error_type,
            'attempt': attempt,
            'max_retries': max_retries,
            'retry_delay_s': round(retry_delay, 2),
        }
        self._queue.put_nowait(log_entry)

    async def _drain_loop(self) -> None:
        """Background task that drains the queue and dispatches to the sink."""
        while True:
            entry = await self._queue.get()
            if entry is None:  # Sentinel — time to shut down
                break
            try:
                self._sink(entry)
            except Exception as e:
                logger.warning(f'LLM call logger sink error: {e}')

    async def _flush(self) -> None:
        """Drain any remaining entries from the queue (non-blocking)."""
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if entry is None:
                continue
            try:
                self._sink(entry)
            except Exception as e:
                logger.warning(f'LLM call logger sink error during flush: {e}')

    @asynccontextmanager
    async def open(self):
        """Activate the logger: open resources and start the background drain task.

        Example — file-based logging::

            logger = LLMCallLogger(log_path="logs/llm_calls.jsonl")
            async with logger.open():
                logger.log_call(model="gpt-4", duration_ms=100, tokens_in=50, tokens_out=20)

        Example — custom sink::

            logger = LLMCallLogger(sink=lambda entry: track_metric("llm_call", entry))
            async with logger.open():
                logger.log_call(model="gpt-4", duration_ms=100, tokens_in=50, tokens_out=20)
        """
        # Open log file if using file sink
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self._log_path, 'a')

        # Start background drain task
        self._drain_task = asyncio.create_task(self._drain_loop())

        try:
            yield self
        finally:
            # Send sentinel to stop drain loop
            self._queue.put_nowait(None)
            if self._drain_task:
                await self._drain_task
                self._drain_task = None

            # Flush any stragglers
            await self._flush()

            # Close file if applicable
            if self._log_file:
                self._log_file.close()
                self._log_file = None
