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
import time

import pytest

from graphiti_core.rate_limiting import (
    RateLimitConfig,
    RateLimiter,
    RateLimitStats,
    ResourceType,
)


class TestRateLimitConfig:
    def test_default_values(self):
        config = RateLimitConfig()
        assert config.max_concurrent_medium == 10
        assert config.max_concurrent_small == 20
        assert config.max_concurrent_embeddings == 50

    def test_custom_values(self):
        config = RateLimitConfig(
            max_concurrent_medium=5,
            max_concurrent_small=15,
            max_concurrent_embeddings=30,
        )
        assert config.max_concurrent_medium == 5
        assert config.max_concurrent_small == 15
        assert config.max_concurrent_embeddings == 30


class TestRateLimitStats:
    def test_default_values(self):
        stats = RateLimitStats()
        for rt in ResourceType:
            assert stats.total_acquisitions[rt] == 0
            assert stats.total_wait_time_ms[rt] == 0.0
            assert stats.current_in_use[rt] == 0


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_default_config(self):
        limiter = RateLimiter()
        assert limiter.config.max_concurrent_medium == 10
        assert limiter.config.max_concurrent_small == 20
        assert limiter.config.max_concurrent_embeddings == 50

    @pytest.mark.asyncio
    async def test_custom_config(self):
        config = RateLimitConfig(max_concurrent_medium=3)
        limiter = RateLimiter(config)
        assert limiter.config.max_concurrent_medium == 3

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        config = RateLimitConfig(max_concurrent_medium=2)
        limiter = RateLimiter(config)

        # Acquire one slot
        wait_time = await limiter.acquire(ResourceType.LLM_MEDIUM)
        assert wait_time >= 0
        stats = limiter.get_stats()
        assert stats.total_acquisitions[ResourceType.LLM_MEDIUM] == 1
        assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 1

        # Release it
        await limiter.release(ResourceType.LLM_MEDIUM)
        stats = limiter.get_stats()
        assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        config = RateLimitConfig(max_concurrent_medium=2)
        limiter = RateLimiter(config)

        async with limiter.limit(ResourceType.LLM_MEDIUM) as wait_time:
            assert wait_time >= 0
            stats = limiter.get_stats()
            assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 1

        # After context, slot should be released
        stats = limiter.get_stats()
        assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 0

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        config = RateLimitConfig(max_concurrent_medium=2)
        limiter = RateLimiter(config)

        with pytest.raises(ValueError):
            async with limiter.limit(ResourceType.LLM_MEDIUM):
                stats = limiter.get_stats()
                assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 1
                raise ValueError('Test error')

        # After context with exception, slot should still be released
        stats = limiter.get_stats()
        assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 0

    @pytest.mark.asyncio
    async def test_semaphore_limit_enforced(self):
        config = RateLimitConfig(max_concurrent_medium=2)
        limiter = RateLimiter(config)

        acquired = []
        start_times = []

        async def worker(worker_id: int, delay: float):
            start_times.append((worker_id, time.time()))
            async with limiter.limit(ResourceType.LLM_MEDIUM):
                acquired.append(worker_id)
                await asyncio.sleep(delay)

        # Start 4 workers with limit of 2
        # Workers should execute in pairs
        start = time.time()
        await asyncio.gather(
            worker(1, 0.05),
            worker(2, 0.05),
            worker(3, 0.05),
            worker(4, 0.05),
        )
        elapsed = time.time() - start

        # With limit of 2 and 4 tasks each taking 0.05s,
        # total time should be at least 0.1s (2 batches)
        assert elapsed >= 0.09  # Allow small margin for timing
        assert len(acquired) == 4

    @pytest.mark.asyncio
    async def test_different_resource_types_independent(self):
        config = RateLimitConfig(
            max_concurrent_medium=1,
            max_concurrent_small=1,
            max_concurrent_embeddings=1,
        )
        limiter = RateLimiter(config)

        # Acquire all three resource types simultaneously
        # This should work because they have independent semaphores
        await limiter.acquire(ResourceType.LLM_MEDIUM)
        await limiter.acquire(ResourceType.LLM_SMALL)
        await limiter.acquire(ResourceType.EMBEDDING)

        stats = limiter.get_stats()
        assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 1
        assert stats.current_in_use[ResourceType.LLM_SMALL] == 1
        assert stats.current_in_use[ResourceType.EMBEDDING] == 1

        # Clean up
        await limiter.release(ResourceType.LLM_MEDIUM)
        await limiter.release(ResourceType.LLM_SMALL)
        await limiter.release(ResourceType.EMBEDDING)

    @pytest.mark.asyncio
    async def test_get_available_slots(self):
        config = RateLimitConfig(max_concurrent_medium=3)
        limiter = RateLimiter(config)

        assert limiter.get_available_slots(ResourceType.LLM_MEDIUM) == 3

        await limiter.acquire(ResourceType.LLM_MEDIUM)
        assert limiter.get_available_slots(ResourceType.LLM_MEDIUM) == 2

        await limiter.acquire(ResourceType.LLM_MEDIUM)
        assert limiter.get_available_slots(ResourceType.LLM_MEDIUM) == 1

        await limiter.release(ResourceType.LLM_MEDIUM)
        assert limiter.get_available_slots(ResourceType.LLM_MEDIUM) == 2

        await limiter.release(ResourceType.LLM_MEDIUM)
        assert limiter.get_available_slots(ResourceType.LLM_MEDIUM) == 3

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        config = RateLimitConfig(max_concurrent_medium=2)
        limiter = RateLimiter(config)

        # Multiple acquire/release cycles
        for _ in range(5):
            await limiter.acquire(ResourceType.LLM_MEDIUM)
            await limiter.release(ResourceType.LLM_MEDIUM)

        stats = limiter.get_stats()
        assert stats.total_acquisitions[ResourceType.LLM_MEDIUM] == 5
        assert stats.current_in_use[ResourceType.LLM_MEDIUM] == 0
        # Wait time should be accumulated (though might be very small)
        assert stats.total_wait_time_ms[ResourceType.LLM_MEDIUM] >= 0

    @pytest.mark.asyncio
    async def test_get_stats_returns_copy(self):
        limiter = RateLimiter()
        stats1 = limiter.get_stats()
        stats2 = limiter.get_stats()

        # Modifying one stats object shouldn't affect the other
        stats1.total_acquisitions[ResourceType.LLM_MEDIUM] = 999
        assert stats2.total_acquisitions[ResourceType.LLM_MEDIUM] == 0
