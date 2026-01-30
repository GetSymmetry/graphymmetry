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

from graphiti_core.llm_client.config import ModelSize
from graphiti_core.metrics import (
    BatchMetrics,
    EpisodeMetrics,
    MetricsCollector,
)


class TestEpisodeMetrics:
    def test_default_values(self):
        metrics = EpisodeMetrics()
        assert metrics.episode_id == ''
        assert metrics.wall_clock_ms == 0.0
        assert metrics.llm_calls_medium == 0
        assert metrics.llm_calls_small == 0
        assert metrics.embedding_calls == 0
        assert metrics.entities_extracted == 0
        assert metrics.edges_extracted == 0
        assert metrics.retries == 0
        assert metrics.errors == []

    def test_custom_values(self):
        metrics = EpisodeMetrics(
            episode_id='test-123',
            wall_clock_ms=1500.5,
            llm_calls_medium=10,
            llm_calls_small=20,
            embedding_calls=5,
            entities_extracted=15,
            edges_extracted=25,
            retries=2,
            errors=['error1', 'error2'],
        )
        assert metrics.episode_id == 'test-123'
        assert metrics.wall_clock_ms == 1500.5
        assert metrics.llm_calls_medium == 10
        assert metrics.llm_calls_small == 20
        assert metrics.embedding_calls == 5
        assert metrics.entities_extracted == 15
        assert metrics.edges_extracted == 25
        assert metrics.retries == 2
        assert metrics.errors == ['error1', 'error2']

    def test_total_llm_calls_property(self):
        metrics = EpisodeMetrics(llm_calls_medium=10, llm_calls_small=20)
        assert metrics.total_llm_calls == 30


class TestBatchMetrics:
    def test_default_values(self):
        metrics = BatchMetrics()
        assert metrics.total_episodes == 0
        assert metrics.completed_episodes == 0
        assert metrics.failed_episodes == 0
        assert metrics.total_wall_clock_ms == 0.0
        assert metrics.total_llm_calls_medium == 0
        assert metrics.total_llm_calls_small == 0
        assert metrics.total_embedding_calls == 0
        assert metrics.total_entities == 0
        assert metrics.total_edges == 0
        assert metrics.total_retries == 0
        assert metrics.episode_metrics == []

    def test_total_llm_calls_property(self):
        metrics = BatchMetrics(total_llm_calls_medium=10, total_llm_calls_small=20)
        assert metrics.total_llm_calls == 30

    def test_add_episode_metrics_success(self):
        batch = BatchMetrics(total_episodes=3)
        episode = EpisodeMetrics(
            episode_id='ep1',
            llm_calls_medium=5,
            llm_calls_small=10,
            embedding_calls=3,
            entities_extracted=8,
            edges_extracted=12,
            retries=1,
        )

        batch.add_episode_metrics(episode, success=True)

        assert batch.completed_episodes == 1
        assert batch.failed_episodes == 0
        assert batch.total_llm_calls_medium == 5
        assert batch.total_llm_calls_small == 10
        assert batch.total_embedding_calls == 3
        assert batch.total_entities == 8
        assert batch.total_edges == 12
        assert batch.total_retries == 1
        assert len(batch.episode_metrics) == 1
        assert batch.episode_metrics[0] == episode

    def test_add_episode_metrics_failure(self):
        batch = BatchMetrics(total_episodes=3)
        episode = EpisodeMetrics(
            episode_id='ep1',
            llm_calls_medium=2,
            retries=3,
            errors=['some error'],
        )

        batch.add_episode_metrics(episode, success=False)

        assert batch.completed_episodes == 0
        assert batch.failed_episodes == 1
        assert batch.total_retries == 3
        assert len(batch.episode_metrics) == 1

    def test_add_multiple_episode_metrics(self):
        batch = BatchMetrics(total_episodes=3)

        batch.add_episode_metrics(
            EpisodeMetrics(llm_calls_medium=5, embedding_calls=2, entities_extracted=3),
            success=True,
        )
        batch.add_episode_metrics(
            EpisodeMetrics(llm_calls_medium=3, embedding_calls=1, entities_extracted=2),
            success=True,
        )
        batch.add_episode_metrics(
            EpisodeMetrics(llm_calls_medium=2, retries=2),
            success=False,
        )

        assert batch.completed_episodes == 2
        assert batch.failed_episodes == 1
        assert batch.total_llm_calls_medium == 10
        assert batch.total_embedding_calls == 3
        assert batch.total_entities == 5
        assert batch.total_retries == 2
        assert len(batch.episode_metrics) == 3


class TestMetricsCollector:
    def test_initialization(self):
        collector = MetricsCollector(episode_id='test-123')
        assert collector.episode_id == 'test-123'

    def test_start_stop_timer(self):
        collector = MetricsCollector()
        collector.start()
        time.sleep(0.05)  # 50ms
        collector.stop()

        metrics = collector.get_metrics()
        assert metrics.wall_clock_ms >= 45  # Allow some margin
        assert metrics.wall_clock_ms < 200  # But not too long

    @pytest.mark.asyncio
    async def test_record_llm_call_medium(self):
        collector = MetricsCollector()
        await collector.record_llm_call(ModelSize.medium)
        await collector.record_llm_call(ModelSize.medium)

        metrics = collector.get_metrics()
        assert metrics.llm_calls_medium == 2
        assert metrics.llm_calls_small == 0

    @pytest.mark.asyncio
    async def test_record_llm_call_small(self):
        collector = MetricsCollector()
        await collector.record_llm_call(ModelSize.small)
        await collector.record_llm_call(ModelSize.small)
        await collector.record_llm_call(ModelSize.small)

        metrics = collector.get_metrics()
        assert metrics.llm_calls_medium == 0
        assert metrics.llm_calls_small == 3

    @pytest.mark.asyncio
    async def test_record_embedding_call(self):
        collector = MetricsCollector()
        await collector.record_embedding_call()
        await collector.record_embedding_call(count=5)

        metrics = collector.get_metrics()
        assert metrics.embedding_calls == 6

    @pytest.mark.asyncio
    async def test_record_retry(self):
        collector = MetricsCollector()
        await collector.record_retry()
        await collector.record_retry()

        metrics = collector.get_metrics()
        assert metrics.retries == 2

    @pytest.mark.asyncio
    async def test_record_error(self):
        collector = MetricsCollector()
        await collector.record_error('Error 1')
        await collector.record_error('Error 2')

        metrics = collector.get_metrics()
        assert metrics.errors == ['Error 1', 'Error 2']

    @pytest.mark.asyncio
    async def test_get_metrics_with_counts(self):
        collector = MetricsCollector(episode_id='test-123')
        collector.start()
        await collector.record_llm_call(ModelSize.medium)
        await collector.record_embedding_call()

        metrics = collector.get_metrics(entities_count=5, edges_count=10)

        assert metrics.episode_id == 'test-123'
        assert metrics.llm_calls_medium == 1
        assert metrics.embedding_calls == 1
        assert metrics.entities_extracted == 5
        assert metrics.edges_extracted == 10

    @pytest.mark.asyncio
    async def test_finalize(self):
        collector = MetricsCollector(episode_id='test-123')
        collector.start()
        time.sleep(0.02)
        await collector.record_llm_call(ModelSize.small)

        metrics = collector.finalize(entities_count=3, edges_count=7)

        assert metrics.episode_id == 'test-123'
        assert metrics.wall_clock_ms >= 15
        assert metrics.llm_calls_small == 1
        assert metrics.entities_extracted == 3
        assert metrics.edges_extracted == 7

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with MetricsCollector(episode_id='ctx-test') as collector:
            time.sleep(0.02)
            await collector.record_llm_call(ModelSize.medium)

        metrics = collector.get_metrics()
        assert metrics.wall_clock_ms >= 15
        assert metrics.llm_calls_medium == 1

    @pytest.mark.asyncio
    async def test_context_manager_records_error_on_exception(self):
        collector = MetricsCollector()

        with pytest.raises(ValueError):
            async with collector:
                raise ValueError('Test error')

        metrics = collector.get_metrics()
        assert 'Test error' in metrics.errors

    @pytest.mark.asyncio
    async def test_concurrent_recording(self):
        """Test that concurrent updates are handled correctly."""
        collector = MetricsCollector()

        async def record_calls(n: int):
            for _ in range(n):
                await collector.record_llm_call(ModelSize.medium)
                await collector.record_embedding_call()

        # Run 10 concurrent tasks each recording 10 calls
        await asyncio.gather(*[record_calls(10) for _ in range(10)])

        metrics = collector.get_metrics()
        assert metrics.llm_calls_medium == 100
        assert metrics.embedding_calls == 100

    def test_get_metrics_without_start(self):
        """Test that get_metrics works even if timer was never started."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert metrics.wall_clock_ms == 0.0

    def test_wall_clock_ongoing(self):
        """Test that get_metrics returns current duration if stop() not called."""
        collector = MetricsCollector()
        collector.start()
        time.sleep(0.03)

        # Get metrics without stopping - should still calculate duration
        metrics = collector.get_metrics()
        assert metrics.wall_clock_ms >= 25

        time.sleep(0.02)
        # Duration should have increased
        metrics2 = collector.get_metrics()
        assert metrics2.wall_clock_ms > metrics.wall_clock_ms
