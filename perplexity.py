"""Pipe for handling Open WebUI requests to perplexity models hosted on LiteLLM.

We are using these pipes to be able to display citations in the frontend.
The standard OpenAI API does not support this.
Thus, use the LiteLLM-native endpoints which return search results and emit them
as required by Open WebUI.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
import httpx
import os
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Pipe:
    """Pipeline for handling Litellm API requests and streaming responses."""

    class Valves(BaseModel):
        """Configuration for the pipeline valves."""

        PERPLEXITY_API_URL: str = Field(
            default_factory=lambda: os.environ.get(
                "LITELLM_API_URL", "https://litellm.ai.viadee.cloud"
            ),
            description="Base URL for the Litellm API.",
        )
        PERPLEXITY_API_KEY: str = Field(
            default_factory=lambda: os.environ.get("LITELLM_KEY", ""),
            description="API key for accessing the Litellm API.",
        )
        MODEL_NAME: str = Field(
            default="perplexity-sonar",
            description="Specifies the model to use for the Litellm API.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.api_url = (
            self.valves.PERPLEXITY_API_URL.rstrip("/") + "/v1/chat/completions"
        )
        self.api_key = self.valves.PERPLEXITY_API_KEY
        self.model_name = self.valves.MODEL_NAME

        if not all([self.api_url, self.api_key, self.model_name]):
            raise ValueError("API URL, API Key, and Model Name must be set.")

    async def emit_search_results(
        self,
        search_results: list[dict],
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ) -> None:
        """
        Emit the search_results payload as 'citation' events.
        """
        for idx, search_result in enumerate(search_results):
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "document": [search_result.get("snippet", "")],
                        "metadata": [
                            {
                                "source": f"document_{idx}",
                                "document_id": idx,
                            }
                        ],
                        "source": {
                            "name": search_result.get("title", "Unknown Source"),
                            "url": search_result.get("url", "Unknown URL"),
                        },
                    },
                }
            )

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict,
        __metadata__: dict,
    ) -> AsyncIterator[str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }
        citations_emitted = False
        messages = body.get("messages", [])
        if not messages:
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {"type": "error", "content": "No messages provided"},
                }
            )
            return

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            # Disable caching as this breaks the return of citations with LiteLLM
            "cache": {
                "no-cache": True,
                "no-store": True,
            },
        }
        for param in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            if param in body:
                payload[param] = body[param]

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", self.api_url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()

                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk.replace("\r\n", "\n")

                        while "\n\n" in buffer:
                            record, buffer = buffer.split("\n\n", 1)
                            for line in record.splitlines():
                                if not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    return
                                try:
                                    chunk_data = json.loads(data_str)
                                    search_results = chunk_data.get(
                                        "search_results", []
                                    )
                                    if not citations_emitted and search_results:
                                        citations_emitted = True
                                        await self.emit_search_results(
                                            search_results, __event_emitter__
                                        )

                                    choices = chunk_data.get("choices")
                                    if choices and isinstance(choices, list):
                                        choice = choices[0]
                                        content = choice.get("delta", {}).get("content")
                                        if content:
                                            yield content
                                            await __event_emitter__(
                                                {
                                                    "type": "progress",
                                                    "data": {
                                                        "content_received": content
                                                    },
                                                }
                                            )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON chunk: {data_str}",
                                        exc_info=True,
                                    )
                                    await __event_emitter__(
                                        {
                                            "type": "notification",
                                            "data": {
                                                "type": "warning",
                                                "content": f"Failed to parse JSON chunk: {data_str}",
                                            },
                                        }
                                    )

        except httpx.RequestError as e:
            logger.exception("Network error during streaming")
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {"type": "error", "content": f"Network error: {e}"},
                }
            )
        except httpx.HTTPStatusError as e:
            logger.exception("HTTP error during streaming")
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "content": f"HTTP error: {e.response.status_code}",
                    },
                }
            )
        except Exception as e:
            logger.exception("Unexpected error in streaming")
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {"type": "error", "content": f"Streaming error: {e}"},
                }
            )
