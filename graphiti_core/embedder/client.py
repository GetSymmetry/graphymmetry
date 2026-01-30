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

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..metrics import MetricsCollector
    from ..rate_limiting import RateLimiter

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))


class EmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)


class EmbedderClient(ABC):
    def __init__(self) -> None:
        self.rate_limiter: 'RateLimiter | None' = None
        self.metrics_collector: 'MetricsCollector | None' = None

    def set_rate_limiter(self, rate_limiter: 'RateLimiter | None') -> None:
        """Set the rate limiter for this embedder client.

        Args:
            rate_limiter: The rate limiter to use, or None to disable rate limiting.
        """
        self.rate_limiter = rate_limiter

    def set_metrics_collector(self, collector: 'MetricsCollector | None') -> None:
        """Set the metrics collector for this embedder client.

        Args:
            collector: The metrics collector to use, or None to disable metrics collection.
        """
        self.metrics_collector = collector

    @abstractmethod
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        pass

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        raise NotImplementedError()

    async def create_with_rate_limit(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embeddings with rate limiting and metrics applied.

        This method wraps the create method with rate limiting if a rate limiter is configured,
        and records metrics if a metrics collector is set.

        Args:
            input_data: The input data to create embeddings for.

        Returns:
            The embedding vector.
        """
        if self.rate_limiter is not None:
            from ..rate_limiting import ResourceType

            async with self.rate_limiter.limit(ResourceType.EMBEDDING):
                result = await self.create(input_data)
        else:
            result = await self.create(input_data)

        # Record metrics if collector is set
        if self.metrics_collector is not None:
            await self.metrics_collector.record_embedding_call()

        return result

    async def create_batch_with_rate_limit(self, input_data_list: list[str]) -> list[list[float]]:
        """Create batch embeddings with rate limiting and metrics applied.

        This method wraps the create_batch method with rate limiting if a rate limiter is configured,
        and records metrics if a metrics collector is set.

        Args:
            input_data_list: The list of input strings to create embeddings for.

        Returns:
            A list of embedding vectors.
        """
        if self.rate_limiter is not None:
            from ..rate_limiting import ResourceType

            async with self.rate_limiter.limit(ResourceType.EMBEDDING):
                result = await self.create_batch(input_data_list)
        else:
            result = await self.create_batch(input_data_list)

        # Record metrics if collector is set (count as 1 batch call)
        if self.metrics_collector is not None:
            await self.metrics_collector.record_embedding_call()

        return result
