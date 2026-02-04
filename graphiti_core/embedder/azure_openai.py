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

import logging
import time
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .client import EmbedderClient

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedderClient(EmbedderClient):
    """Wrapper class for Azure OpenAI that implements the EmbedderClient interface.

    Supports both AsyncAzureOpenAI and AsyncOpenAI (with Azure v1 API endpoint).
    """

    def __init__(
        self,
        azure_client: AsyncAzureOpenAI | AsyncOpenAI,
        model: str = 'text-embedding-3-small',
    ):
        super().__init__()
        self.azure_client = azure_client
        self.model = model

    async def _create_impl(self, input_data: str | list[str] | Any) -> list[float]:
        """Create embeddings using Azure OpenAI client."""
        start_time = time.time()
        try:
            # Handle different input types
            if isinstance(input_data, str):
                text_input = [input_data]
            elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                text_input = input_data
            else:
                # Convert to string list for other types
                text_input = [str(input_data)]

            response = await self.azure_client.embeddings.create(model=self.model, input=text_input)

            # Log tokens if call logger is set (embeddings only have input tokens)
            if self._call_logger is not None:
                usage = getattr(response, 'usage', None)
                if usage is not None:
                    self._call_logger.log_call(
                        model=self.model,
                        duration_ms=(time.time() - start_time) * 1000,
                        tokens_in=getattr(usage, 'prompt_tokens', 0) or 0,
                        tokens_out=0,  # Embeddings don't have output tokens
                    )

            # Return the first embedding as a list of floats
            return response.data[0].embedding
        except Exception as e:
            logger.error(f'Error in Azure OpenAI embedding: {e}')
            raise

    async def _create_batch_impl(self, input_data_list: list[str]) -> list[list[float]]:
        """Create batch embeddings using Azure OpenAI client."""
        start_time = time.time()
        try:
            response = await self.azure_client.embeddings.create(
                model=self.model, input=input_data_list
            )

            # Log tokens if call logger is set (embeddings only have input tokens)
            if self._call_logger is not None:
                usage = getattr(response, 'usage', None)
                if usage is not None:
                    self._call_logger.log_call(
                        model=self.model,
                        duration_ms=(time.time() - start_time) * 1000,
                        tokens_in=getattr(usage, 'prompt_tokens', 0) or 0,
                        tokens_out=0,  # Embeddings don't have output tokens
                    )

            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f'Error in Azure OpenAI batch embedding: {e}')
            raise
