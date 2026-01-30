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
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client.config import ModelSize

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics collected during episode processing.

    Attributes:
        episode_id: UUID of the episode these metrics are for.
        wall_clock_ms: Total wall clock time for processing the episode.
        llm_calls_medium: Number of LLM calls using medium/large models.
        llm_calls_small: Number of LLM calls using small models.
        embedding_calls: Number of embedding API calls.
        entities_extracted: Number of entity nodes extracted.
        edges_extracted: Number of entity edges extracted.
        retries: Total number of retries across all API calls.
        errors: List of error messages encountered during processing.
    """

    episode_id: str = ''
    wall_clock_ms: float = 0.0
    llm_calls_medium: int = 0
    llm_calls_small: int = 0
    embedding_calls: int = 0
    entities_extracted: int = 0
    edges_extracted: int = 0
    retries: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_llm_calls(self) -> int:
        """Total LLM calls (medium + small)."""
        return self.llm_calls_medium + self.llm_calls_small


@dataclass
class BatchMetrics:
    """Aggregated metrics across a batch of episodes.

    Attributes:
        total_episodes: Total number of episodes in the batch.
        completed_episodes: Number of episodes successfully processed.
        failed_episodes: Number of episodes that failed processing.
        total_wall_clock_ms: Total wall clock time for the entire batch.
        total_llm_calls_medium: Total medium/large model LLM calls.
        total_llm_calls_small: Total small model LLM calls.
        total_embedding_calls: Total embedding API calls.
        total_entities: Total entities extracted across all episodes.
        total_edges: Total edges extracted across all episodes.
        total_retries: Total retries across all API calls.
        episode_metrics: Per-episode metrics for detailed analysis.
    """

    total_episodes: int = 0
    completed_episodes: int = 0
    failed_episodes: int = 0
    total_wall_clock_ms: float = 0.0
    total_llm_calls_medium: int = 0
    total_llm_calls_small: int = 0
    total_embedding_calls: int = 0
    total_entities: int = 0
    total_edges: int = 0
    total_retries: int = 0
    episode_metrics: list[EpisodeMetrics] = field(default_factory=list)

    @property
    def total_llm_calls(self) -> int:
        """Total LLM calls (medium + small)."""
        return self.total_llm_calls_medium + self.total_llm_calls_small

    def add_episode_metrics(self, metrics: EpisodeMetrics, success: bool = True) -> None:
        """Add episode metrics to the batch aggregation.

        Args:
            metrics: The episode metrics to add.
            success: Whether the episode completed successfully.
        """
        self.episode_metrics.append(metrics)
        if success:
            self.completed_episodes += 1
        else:
            self.failed_episodes += 1
        self.total_llm_calls_medium += metrics.llm_calls_medium
        self.total_llm_calls_small += metrics.llm_calls_small
        self.total_embedding_calls += metrics.embedding_calls
        self.total_entities += metrics.entities_extracted
        self.total_edges += metrics.edges_extracted
        self.total_retries += metrics.retries


class MetricsCollector:
    """Collects metrics during episode processing.

    This class is thread-safe and can be shared across concurrent async operations.
    It tracks LLM calls, embedding calls, retries, and errors.

    Usage:
        collector = MetricsCollector(episode_id="abc123")
        collector.start()

        # Pass to LLM/embedder clients
        llm_client.set_metrics_collector(collector)

        # ... process episode ...

        metrics = collector.finalize(entities_count=5, edges_count=3)

    Or as a context manager:
        async with MetricsCollector(episode_id="abc123") as collector:
            # ... process episode ...
            metrics = collector.get_metrics(entities_count=5, edges_count=3)
    """

    def __init__(self, episode_id: str = ''):
        """Initialize the metrics collector.

        Args:
            episode_id: UUID of the episode being processed.
        """
        self.episode_id = episode_id
        self._llm_calls_medium = 0
        self._llm_calls_small = 0
        self._embedding_calls = 0
        self._retries = 0
        self._errors: list[str] = []
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._lock = asyncio.Lock()

    def start(self) -> None:
        """Start the wall clock timer."""
        self._start_time = perf_counter()

    def stop(self) -> None:
        """Stop the wall clock timer."""
        self._end_time = perf_counter()

    async def record_llm_call(self, model_size: 'ModelSize') -> None:
        """Record an LLM API call.

        Args:
            model_size: The size of the model used (small or medium).
        """
        from .llm_client.config import ModelSize

        async with self._lock:
            if model_size == ModelSize.small:
                self._llm_calls_small += 1
            else:
                self._llm_calls_medium += 1

    async def record_embedding_call(self, count: int = 1) -> None:
        """Record embedding API call(s).

        Args:
            count: Number of embedding calls to record (for batch operations).
        """
        async with self._lock:
            self._embedding_calls += count

    async def record_retry(self) -> None:
        """Record an API retry."""
        async with self._lock:
            self._retries += 1

    async def record_error(self, error: str) -> None:
        """Record an error message.

        Args:
            error: The error message to record.
        """
        async with self._lock:
            self._errors.append(error)

    def get_metrics(
        self,
        entities_count: int = 0,
        edges_count: int = 0,
    ) -> EpisodeMetrics:
        """Get the collected metrics.

        Args:
            entities_count: Number of entities extracted.
            edges_count: Number of edges extracted.

        Returns:
            EpisodeMetrics with all collected data.
        """
        wall_clock_ms = 0.0
        if self._start_time is not None:
            end = self._end_time if self._end_time is not None else perf_counter()
            wall_clock_ms = (end - self._start_time) * 1000

        return EpisodeMetrics(
            episode_id=self.episode_id,
            wall_clock_ms=wall_clock_ms,
            llm_calls_medium=self._llm_calls_medium,
            llm_calls_small=self._llm_calls_small,
            embedding_calls=self._embedding_calls,
            entities_extracted=entities_count,
            edges_extracted=edges_count,
            retries=self._retries,
            errors=list(self._errors),
        )

    def finalize(
        self,
        entities_count: int = 0,
        edges_count: int = 0,
    ) -> EpisodeMetrics:
        """Stop the timer and return final metrics.

        Args:
            entities_count: Number of entities extracted.
            edges_count: Number of edges extracted.

        Returns:
            EpisodeMetrics with all collected data.
        """
        self.stop()
        return self.get_metrics(entities_count, edges_count)

    async def __aenter__(self) -> 'MetricsCollector':
        """Async context manager entry - starts the timer."""
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - stops the timer."""
        self.stop()
        if exc_val is not None:
            await self.record_error(str(exc_val))
