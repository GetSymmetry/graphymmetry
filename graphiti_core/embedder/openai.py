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

import time
from collections.abc import Iterable

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'


class OpenAIEmbedderConfig(EmbedderConfig):
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None


class OpenAIEmbedder(EmbedderClient):
    """
    OpenAI Embedder Client

    This client supports both AsyncOpenAI and AsyncAzureOpenAI clients.
    """

    def __init__(
        self,
        config: OpenAIEmbedderConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        super().__init__()
        if config is None:
            config = OpenAIEmbedderConfig()
        self.config = config

        if client is not None:
            self.client = client
        else:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def _create_impl(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        start_time = time.time()
        result = await self.client.embeddings.create(
            input=input_data, model=self.config.embedding_model
        )

        # Log tokens if call logger is set (embeddings only have input tokens)
        if self._call_logger is not None:
            usage = getattr(result, 'usage', None)
            if usage is not None:
                self._call_logger.log_call(
                    model=str(self.config.embedding_model),
                    duration_ms=(time.time() - start_time) * 1000,
                    tokens_in=getattr(usage, 'prompt_tokens', 0) or 0,
                    tokens_out=0,  # Embeddings don't have output tokens
                )

        return result.data[0].embedding[: self.config.embedding_dim]

    async def _create_batch_impl(self, input_data_list: list[str]) -> list[list[float]]:
        start_time = time.time()
        result = await self.client.embeddings.create(
            input=input_data_list, model=self.config.embedding_model
        )

        # Log tokens if call logger is set (embeddings only have input tokens)
        if self._call_logger is not None:
            usage = getattr(result, 'usage', None)
            if usage is not None:
                self._call_logger.log_call(
                    model=str(self.config.embedding_model),
                    duration_ms=(time.time() - start_time) * 1000,
                    tokens_in=getattr(usage, 'prompt_tokens', 0) or 0,
                    tokens_out=0,  # Embeddings don't have output tokens
                )

        return [embedding.embedding[: self.config.embedding_dim] for embedding in result.data]
