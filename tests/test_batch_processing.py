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
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.graphiti import (
    AddEpisodeResults,
    BatchResults,
    EpisodeInput,
    ErrorAction,
    Graphiti,
)
from graphiti_core.metrics import BatchMetrics, EpisodeMetrics
from graphiti_core.nodes import EpisodeType, EpisodicNode


class TestEpisodeInput:
    def test_required_fields(self):
        episode = EpisodeInput(
            name='test_episode',
            episode_body='Hello, world!',
            source_description='Test source',
            reference_time=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert episode.name == 'test_episode'
        assert episode.episode_body == 'Hello, world!'
        assert episode.source_description == 'Test source'
        assert episode.reference_time == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_default_values(self):
        episode = EpisodeInput(
            name='test',
            episode_body='body',
            source_description='desc',
            reference_time=datetime.now(timezone.utc),
        )
        assert episode.source == EpisodeType.message
        assert episode.group_id is None
        assert episode.uuid is None
        assert episode.update_communities is False
        assert episode.entity_types is None
        assert episode.excluded_entity_types is None
        assert episode.edge_types is None
        assert episode.edge_type_map is None
        assert episode.custom_extraction_instructions is None
        assert episode.saga is None
        assert episode.saga_previous_episode_uuid is None

    def test_custom_values(self):
        ref_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        episode = EpisodeInput(
            name='custom_episode',
            episode_body='Custom body',
            source_description='Custom source',
            reference_time=ref_time,
            source=EpisodeType.text,
            group_id='my-group',
            uuid='custom-uuid',
            update_communities=True,
            saga='my-saga',
        )
        assert episode.source == EpisodeType.text
        assert episode.group_id == 'my-group'
        assert episode.uuid == 'custom-uuid'
        assert episode.update_communities is True
        assert episode.saga == 'my-saga'


class TestErrorAction:
    def test_abort_batch_value(self):
        assert ErrorAction.ABORT_BATCH.value == 'abort'

    def test_skip_episode_value(self):
        assert ErrorAction.SKIP_EPISODE.value == 'skip'


class TestBatchResults:
    def test_default_values(self):
        results = BatchResults()
        assert results.successful == []
        assert results.failed == []
        assert isinstance(results.metrics, BatchMetrics)

    def test_custom_metrics(self):
        metrics = BatchMetrics(total_episodes=5)
        results = BatchResults(metrics=metrics)
        assert results.metrics.total_episodes == 5


class TestAddEpisodesBatch:
    @pytest.fixture
    def mock_graphiti(self):
        """Create a mock Graphiti instance."""
        with patch.object(Graphiti, '__init__', lambda self, **kwargs: None):
            graphiti = Graphiti()
            graphiti.llm_client = MagicMock()
            graphiti.embedder = MagicMock()
            graphiti.driver = MagicMock()
            graphiti.rate_limiter = None
            return graphiti

    def create_mock_result(self, episode_id: str = 'ep-123') -> AddEpisodeResults:
        """Create a mock AddEpisodeResults."""
        return AddEpisodeResults(
            episode=EpisodicNode(
                uuid=episode_id,
                name='test',
                source=EpisodeType.message,
                content='test content',
                source_description='test',
                group_id='default',
                valid_at=datetime.now(timezone.utc),
                entity_edges=[],
            ),
            episodic_edges=[],
            nodes=[],
            edges=[],
            communities=[],
            community_edges=[],
            metrics=EpisodeMetrics(
                episode_id=episode_id,
                wall_clock_ms=100.0,
                llm_calls_medium=5,
                llm_calls_small=3,
                embedding_calls=2,
                entities_extracted=4,
                edges_extracted=6,
            ),
        )

    @pytest.mark.asyncio
    async def test_empty_episodes_list(self, mock_graphiti):
        """Test with empty episodes list."""
        results = await mock_graphiti.add_episodes_batch(episodes=[])

        assert results.successful == []
        assert results.failed == []
        assert results.metrics.total_episodes == 0
        assert results.metrics.completed_episodes == 0
        assert results.metrics.total_wall_clock_ms > 0

    @pytest.mark.asyncio
    async def test_single_episode_success(self, mock_graphiti):
        """Test successful processing of a single episode."""
        mock_result = self.create_mock_result('ep-1')
        mock_graphiti.add_episode = AsyncMock(return_value=mock_result)

        episode = EpisodeInput(
            name='episode-1',
            episode_body='Test body',
            source_description='Test source',
            reference_time=datetime.now(timezone.utc),
        )

        results = await mock_graphiti.add_episodes_batch(episodes=[episode])

        assert len(results.successful) == 1
        assert len(results.failed) == 0
        assert results.metrics.completed_episodes == 1
        assert results.metrics.failed_episodes == 0
        assert results.metrics.total_llm_calls_medium == 5
        mock_graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_episodes_success(self, mock_graphiti):
        """Test successful processing of multiple episodes."""
        call_count = 0

        async def mock_add_episode(**kwargs):
            nonlocal call_count
            call_count += 1
            return self.create_mock_result(f'ep-{call_count}')

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name=f'episode-{i}',
                episode_body=f'Body {i}',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        results = await mock_graphiti.add_episodes_batch(episodes=episodes)

        assert len(results.successful) == 3
        assert len(results.failed) == 0
        assert results.metrics.completed_episodes == 3
        assert results.metrics.total_llm_calls_medium == 15  # 5 per episode

    @pytest.mark.asyncio
    async def test_episode_failure_skip(self, mock_graphiti):
        """Test that failures are recorded when error handler returns SKIP_EPISODE."""
        call_count = 0

        async def mock_add_episode(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError('Simulated failure')
            return self.create_mock_result(f'ep-{call_count}')

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name=f'episode-{i}',
                episode_body=f'Body {i}',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        error_handler_called = []

        def on_error(name, error):
            error_handler_called.append((name, str(error)))
            return ErrorAction.SKIP_EPISODE

        results = await mock_graphiti.add_episodes_batch(
            episodes=episodes,
            on_error=on_error,
        )

        assert len(results.successful) == 2
        assert len(results.failed) == 1
        assert results.metrics.completed_episodes == 2
        assert results.metrics.failed_episodes == 1
        assert len(error_handler_called) == 1
        assert 'Simulated failure' in error_handler_called[0][1]

    @pytest.mark.asyncio
    async def test_episode_failure_abort(self, mock_graphiti):
        """Test that processing stops when error handler returns ABORT_BATCH."""
        call_order = []

        async def mock_add_episode(**kwargs):
            name = kwargs['name']
            call_order.append(name)
            # Small delay to ensure order
            await asyncio.sleep(0.01)
            if name == 'episode-0':
                raise ValueError('Abort trigger')
            return self.create_mock_result(name)

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name=f'episode-{i}',
                episode_body=f'Body {i}',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
            for i in range(5)
        ]

        def on_error(name, error):
            return ErrorAction.ABORT_BATCH

        # With parallelism=1, we control execution order
        results = await mock_graphiti.add_episodes_batch(
            episodes=episodes,
            parallelism=1,
            on_error=on_error,
        )

        # First episode failed, abort should prevent remaining
        assert results.metrics.failed_episodes >= 1
        # Not all episodes should have been processed
        assert results.metrics.completed_episodes + results.metrics.failed_episodes < 5

    @pytest.mark.asyncio
    async def test_on_episode_complete_callback(self, mock_graphiti):
        """Test that on_episode_complete callback is invoked."""
        mock_graphiti.add_episode = AsyncMock(return_value=self.create_mock_result())

        episode = EpisodeInput(
            name='callback-test',
            episode_body='Test',
            source_description='Test',
            reference_time=datetime.now(timezone.utc),
        )

        callback_invocations = []

        def on_complete(name, metrics):
            callback_invocations.append((name, metrics))

        await mock_graphiti.add_episodes_batch(
            episodes=[episode],
            on_episode_complete=on_complete,
        )

        assert len(callback_invocations) == 1
        assert callback_invocations[0][0] == 'callback-test'
        assert isinstance(callback_invocations[0][1], EpisodeMetrics)

    @pytest.mark.asyncio
    async def test_parallelism_enforcement(self, mock_graphiti):
        """Test that parallelism limit is enforced."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_add_episode(**kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            await asyncio.sleep(0.05)  # Simulate work

            async with lock:
                current_concurrent -= 1

            return self.create_mock_result()

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name=f'episode-{i}',
                episode_body=f'Body {i}',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
            for i in range(10)
        ]

        await mock_graphiti.add_episodes_batch(
            episodes=episodes,
            parallelism=3,
        )

        # Max concurrent should not exceed parallelism limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_metrics_aggregation(self, mock_graphiti):
        """Test that metrics are properly aggregated across episodes."""
        call_count = 0

        async def mock_add_episode(**kwargs):
            nonlocal call_count
            call_count += 1
            return AddEpisodeResults(
                episode=EpisodicNode(
                    uuid=f'ep-{call_count}',
                    name='test',
                    source=EpisodeType.message,
                    content='test',
                    source_description='test',
                    group_id='default',
                    valid_at=datetime.now(timezone.utc),
                    entity_edges=[],
                ),
                episodic_edges=[],
                nodes=[],
                edges=[],
                communities=[],
                community_edges=[],
                metrics=EpisodeMetrics(
                    episode_id=f'ep-{call_count}',
                    wall_clock_ms=100.0,
                    llm_calls_medium=call_count,  # Varying values
                    llm_calls_small=call_count * 2,
                    embedding_calls=call_count,
                    entities_extracted=call_count,
                    edges_extracted=call_count,
                    retries=1,
                ),
            )

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name=f'episode-{i}',
                episode_body=f'Body {i}',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        results = await mock_graphiti.add_episodes_batch(episodes=episodes)

        # 1 + 2 + 3 = 6
        assert results.metrics.total_llm_calls_medium == 6
        # 2 + 4 + 6 = 12
        assert results.metrics.total_llm_calls_small == 12
        # 1 + 2 + 3 = 6
        assert results.metrics.total_embedding_calls == 6
        assert results.metrics.total_entities == 6
        assert results.metrics.total_edges == 6
        assert results.metrics.total_retries == 3
        assert len(results.metrics.episode_metrics) == 3

    @pytest.mark.asyncio
    async def test_episode_input_passthrough(self, mock_graphiti):
        """Test that EpisodeInput fields are correctly passed to add_episode."""
        received_kwargs = {}

        async def mock_add_episode(**kwargs):
            received_kwargs.update(kwargs)
            return self.create_mock_result()

        mock_graphiti.add_episode = mock_add_episode

        ref_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        episode = EpisodeInput(
            name='passthrough-test',
            episode_body='Test body content',
            source_description='Test source description',
            reference_time=ref_time,
            source=EpisodeType.text,
            group_id='test-group',
            uuid='test-uuid',
            update_communities=True,
            custom_extraction_instructions='Custom instructions',
            saga='test-saga',
        )

        await mock_graphiti.add_episodes_batch(episodes=[episode])

        assert received_kwargs['name'] == 'passthrough-test'
        assert received_kwargs['episode_body'] == 'Test body content'
        assert received_kwargs['source_description'] == 'Test source description'
        assert received_kwargs['reference_time'] == ref_time
        assert received_kwargs['source'] == EpisodeType.text
        assert received_kwargs['group_id'] == 'test-group'
        assert received_kwargs['uuid'] == 'test-uuid'
        assert received_kwargs['update_communities'] is True
        assert received_kwargs['custom_extraction_instructions'] == 'Custom instructions'
        assert received_kwargs['saga'] == 'test-saga'

    @pytest.mark.asyncio
    async def test_default_error_behavior_skips(self, mock_graphiti):
        """Test that default behavior (no on_error callback) skips failed episodes."""
        call_count = 0

        async def mock_add_episode(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError('Test failure')
            return self.create_mock_result(f'ep-{call_count}')

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name=f'episode-{i}',
                episode_body=f'Body {i}',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        # No on_error callback provided
        results = await mock_graphiti.add_episodes_batch(episodes=episodes)

        # Should continue processing after failure
        assert results.metrics.completed_episodes == 2
        assert results.metrics.failed_episodes == 1

    @pytest.mark.asyncio
    async def test_wall_clock_time_recorded(self, mock_graphiti):
        """Test that total wall clock time is recorded."""

        async def mock_add_episode(**kwargs):
            await asyncio.sleep(0.05)
            return self.create_mock_result()

        mock_graphiti.add_episode = mock_add_episode

        episodes = [
            EpisodeInput(
                name='time-test',
                episode_body='Body',
                source_description='Test',
                reference_time=datetime.now(timezone.utc),
            )
        ]

        results = await mock_graphiti.add_episodes_batch(episodes=episodes)

        # Should have recorded at least 50ms
        assert results.metrics.total_wall_clock_ms >= 45
