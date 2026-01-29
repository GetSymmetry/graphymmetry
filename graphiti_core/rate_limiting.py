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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be rate-limited."""

    LLM_MEDIUM = 'llm_medium'  # Larger models (gpt-4, claude-opus, etc.)
    LLM_SMALL = 'llm_small'  # Smaller models (gpt-4-mini, claude-haiku, etc.)
    EMBEDDING = 'embedding'  # Embedding API calls


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting across LLM and embedding clients.

    This configuration is shared across all clients within a Graphiti instance,
    providing coordinated rate limiting for parallel operations.

    Attributes:
        max_concurrent_medium: Maximum concurrent calls to medium/large LLM models.
        max_concurrent_small: Maximum concurrent calls to small LLM models.
        max_concurrent_embeddings: Maximum concurrent embedding API calls.
    """

    max_concurrent_medium: int = 10
    max_concurrent_small: int = 20
    max_concurrent_embeddings: int = 50


@dataclass
class RateLimitStats:
    """Statistics tracked by the rate limiter.

    Attributes:
        total_acquisitions: Total number of times a resource was acquired.
        total_wait_time_ms: Total time spent waiting for semaphores (milliseconds).
        current_in_use: Dictionary of currently in-use counts per resource type.
    """

    total_acquisitions: dict[ResourceType, int] = field(
        default_factory=lambda: {rt: 0 for rt in ResourceType}
    )
    total_wait_time_ms: dict[ResourceType, float] = field(
        default_factory=lambda: {rt: 0.0 for rt in ResourceType}
    )
    current_in_use: dict[ResourceType, int] = field(
        default_factory=lambda: {rt: 0 for rt in ResourceType}
    )


class RateLimiter:
    """Shared rate limiter with per-resource-type semaphores.

    This class provides coordinated rate limiting across LLM and embedding clients.
    It uses asyncio semaphores to limit concurrent operations and tracks statistics
    for monitoring and debugging.

    Example usage:
        config = RateLimitConfig(max_concurrent_medium=5)
        rate_limiter = RateLimiter(config)

        async with rate_limiter.limit(ResourceType.LLM_MEDIUM):
            result = await llm_client.generate_response(...)
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize the rate limiter with the given configuration.

        Args:
            config: Rate limit configuration. If None, uses default values.
        """
        if config is None:
            config = RateLimitConfig()

        self.config = config
        self._semaphores: dict[ResourceType, asyncio.Semaphore] = {
            ResourceType.LLM_MEDIUM: asyncio.Semaphore(config.max_concurrent_medium),
            ResourceType.LLM_SMALL: asyncio.Semaphore(config.max_concurrent_small),
            ResourceType.EMBEDDING: asyncio.Semaphore(config.max_concurrent_embeddings),
        }
        self._stats = RateLimitStats()
        self._lock = asyncio.Lock()

    async def acquire(self, resource_type: ResourceType) -> float:
        """Acquire a slot for the given resource type.

        This method blocks until a slot is available.

        Args:
            resource_type: The type of resource to acquire.

        Returns:
            The time spent waiting in milliseconds.
        """
        start_time = asyncio.get_event_loop().time()
        semaphore = self._semaphores[resource_type]

        await semaphore.acquire()

        wait_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        async with self._lock:
            self._stats.total_acquisitions[resource_type] += 1
            self._stats.total_wait_time_ms[resource_type] += wait_time_ms
            self._stats.current_in_use[resource_type] += 1

        if wait_time_ms > 100:  # Log if wait was significant
            logger.debug(
                f'Rate limiter: acquired {resource_type.value} after {wait_time_ms:.1f}ms wait'
            )

        return wait_time_ms

    async def release(self, resource_type: ResourceType) -> None:
        """Release a slot for the given resource type.

        Args:
            resource_type: The type of resource to release.
        """
        semaphore = self._semaphores[resource_type]
        semaphore.release()

        async with self._lock:
            self._stats.current_in_use[resource_type] -= 1

    @asynccontextmanager
    async def limit(self, resource_type: ResourceType) -> AsyncIterator[float]:
        """Context manager for rate-limited operations.

        Acquires a slot before entering the context and releases it when exiting,
        even if an exception occurs.

        Args:
            resource_type: The type of resource to rate-limit.

        Yields:
            The time spent waiting for the slot in milliseconds.

        Example:
            async with rate_limiter.limit(ResourceType.LLM_MEDIUM) as wait_time:
                result = await llm_client.generate_response(...)
        """
        wait_time = await self.acquire(resource_type)
        try:
            yield wait_time
        finally:
            await self.release(resource_type)

    def get_stats(self) -> RateLimitStats:
        """Get a copy of the current statistics.

        Returns:
            A copy of the current rate limit statistics.
        """
        return RateLimitStats(
            total_acquisitions=dict(self._stats.total_acquisitions),
            total_wait_time_ms=dict(self._stats.total_wait_time_ms),
            current_in_use=dict(self._stats.current_in_use),
        )

    def get_available_slots(self, resource_type: ResourceType) -> int:
        """Get the number of available slots for a resource type.

        Args:
            resource_type: The type of resource to check.

        Returns:
            The number of available slots.
        """
        semaphore = self._semaphores[resource_type]
        # Semaphore._value gives current available count
        return semaphore._value  # type: ignore[attr-defined]
