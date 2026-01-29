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
    from ..rate_limiting import RateLimiter

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', 1024))


class EmbedderConfig(BaseModel):
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)


class EmbedderClient(ABC):
    def __init__(self) -> None:
        self.rate_limiter: 'RateLimiter | None' = None

    def set_rate_limiter(self, rate_limiter: 'RateLimiter | None') -> None:
        """Set the rate limiter for this embedder client.

        Args:
            rate_limiter: The rate limiter to use, or None to disable rate limiting.
        """
        self.rate_limiter = rate_limiter

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
        """Create embeddings with rate limiting applied.

        This method wraps the create method with rate limiting if a rate limiter is configured.

        Args:
            input_data: The input data to create embeddings for.

        Returns:
            The embedding vector.
        """
        if self.rate_limiter is not None:
            from ..rate_limiting import ResourceType

            async with self.rate_limiter.limit(ResourceType.EMBEDDING):
                return await self.create(input_data)
        return await self.create(input_data)

    async def create_batch_with_rate_limit(self, input_data_list: list[str]) -> list[list[float]]:
        """Create batch embeddings with rate limiting applied.

        This method wraps the create_batch method with rate limiting if a rate limiter is configured.

        Args:
            input_data_list: The list of input strings to create embeddings for.

        Returns:
            A list of embedding vectors.
        """
        if self.rate_limiter is not None:
            from ..rate_limiting import ResourceType

            async with self.rate_limiter.limit(ResourceType.EMBEDDING):
                return await self.create_batch(input_data_list)
        return await self.create_batch(input_data_list)
