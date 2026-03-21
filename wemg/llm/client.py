import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
import openai
import pydantic

from wemg.llm.cache import RedisCacheManager
from wemg.llm.parsing import extract_info_from_text

litellm.drop_params = True
logger = logging.getLogger(__name__)


class LLMClient:
    """Single unified LLM client for completions, embeddings, and reranking.
    Uses litellm for multi-provider support, OpenAI client for direct access,
    Redis caching, and ThreadPoolExecutor for batch parallelism.
    """

    def __init__(
        self,
        model_name: str,
        url: str,
        api_key: Optional[str] = None,
        concurrency: int = 64,
        max_retries: int = 1,
        cache_config: Optional[Dict[str, Any]] = None,
        is_embedding: bool = False,
        **generation_kwargs,
    ):
        self.model_name = model_name
        self.url = url
        self.api_key = api_key
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.is_embedding = is_embedding

        self.timeout = generation_kwargs.get("timeout", 300)
        self.temperature = generation_kwargs.get("temperature", 0.7)
        self.num_samples = generation_kwargs.get("n", 1)
        self.top_p = generation_kwargs.get("top_p", 0.8)
        self.max_tokens = generation_kwargs.get("max_tokens", 8192)
        self.max_input_tokens = generation_kwargs.get("max_input_tokens", 32768)
        self.top_k = generation_kwargs.get("top_k", 20)
        self.enable_thinking = generation_kwargs.get("enable_thinking", True)
        self.random_seed = generation_kwargs.get("random_seed", None)

        self._openai_client = openai.OpenAI(api_key=api_key, base_url=url)

        self._cache: Optional[RedisCacheManager] = None
        self._use_cache = False
        if cache_config:
            cache_config = dict(cache_config)
            if cache_config.pop("enabled", False):
                try:
                    self._cache = RedisCacheManager(**cache_config)
                    self._use_cache = True
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis cache: {e}")

    def _prepare_model_kwargs(self, **kwargs) -> Dict[str, Any]:
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        output_schema = kwargs.get("output_schema", None)

        response_format = None
        if output_schema is not None:
            assert issubclass(output_schema, pydantic.BaseModel)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema(),
                },
            }

        return {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "n": kwargs.get("n", self.num_samples),
            "seed": kwargs.get("random_seed", self.random_seed),
            "response_format": response_format,
            "timeout": kwargs.get("timeout", self.timeout),
            "logprobs": kwargs.get("logprobs", None),
            "top_logprobs": kwargs.get("top_logprobs", None),
            "extra_body": {
                "top_k": kwargs.get("top_k", self.top_k),
                "chat_template_kwargs": {
                    "enable_thinking": True if enable_thinking else None,
                },
            },
        }

    def generate(
        self, index: int, messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[int, List[Dict]]:
        """Generate completion for a single set of messages."""
        should_return_reasoning = kwargs.get("should_return_reasoning", True)
        output_schema = kwargs.get("output_schema", None)
        use_cache = kwargs.get("use_cache", self._use_cache)
        cache_ttl = kwargs.get("cache_ttl", None)

        if output_schema is not None:
            assert issubclass(output_schema, pydantic.BaseModel)

        model_kwargs = self._prepare_model_kwargs(**kwargs)
        n = model_kwargs.get("n", 1)

        cache_key_data = {
            "model": self.model_name,
            "messages": messages,
            "kwargs": {
                k: v for k, v in model_kwargs.items() if k not in ["timeout"]
            },
        }

        if use_cache and self._cache:
            cached_result = self._cache.get(cache_key_data)
            if cached_result is not None:
                logger.info(f"Using cached result for request {index}")
                return index, cached_result

        valid_choices: List[Dict] = []
        attempts = 0
        response = None
        original_n = n

        while len(valid_choices) < original_n and attempts < self.max_retries:
            attempts += 1
            logger.info(
                f"Generating with parameters: {model_kwargs} in attempt {attempts}"
            )
            try:
                messages = litellm.utils.trim_messages(
                    messages, max_tokens=self.max_input_tokens + 1024 * attempts
                )
                response = self._openai_client.chat.completions.create(
                    messages=messages, **model_kwargs
                )
            except Exception as e:
                logger.warning(f"Error during completion: {e}. Retrying...")
                response = None
                time.sleep(2 * attempts)

            if response is None or not hasattr(response, "choices"):
                logger.warning(
                    f"Response is None or does not have choices at attempt {attempts}"
                )
                # Escalate parameters only on failure for the next retry
                model_kwargs["temperature"] = min(
                    1.0, model_kwargs.get("temperature", 0.7) + 0.1
                )
                model_kwargs["top_p"] = min(
                    1.0, model_kwargs.get("top_p", 0.8) + 0.1
                )
                model_kwargs["max_tokens"] = min(
                    model_kwargs.get("max_tokens", self.max_tokens) + 1024,
                    65536,
                )
                continue

            for choice in response.choices:
                if not choice.message:
                    continue

                content = choice.message.content
                used_reasoning_content_as_content = False
                if not content:
                    try:
                        content = choice.message.reasoning_content
                        used_reasoning_content_as_content = True
                    except Exception:
                        logger.warning(
                            f"No content in response for attempt {attempts}"
                        )
                        continue

                if not content:
                    logger.warning(f"No content in response for attempt {attempts}")
                    continue

                reasoning = None
                if should_return_reasoning and not used_reasoning_content_as_content:
                    try:
                        reasoning = choice.message.reasoning_content
                    except Exception:
                        reasoning = None

                output = {
                    "output": content,
                    "raw_output": content,
                    "reasoning": reasoning,
                    "is_valid": True,
                }

                if output_schema:
                    output = None
                    try:
                        parsed_output = output_schema.model_validate_json(content)
                        output = {
                            "output": (
                                parsed_output.model_dump()
                                if hasattr(parsed_output, "model_dump")
                                else parsed_output
                            ),
                            "raw_output": content,
                            "reasoning": reasoning,
                            "is_valid": True,
                        }
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse structured output for "
                            f"{content[:100]}... at attempt {attempts}: {e}"
                        )
                        if attempts == self.max_retries - 1:
                            output = {
                                "output": content,
                                "raw_output": content,
                                "reasoning": reasoning,
                                "is_valid": False,
                            }

                if output is not None:
                    valid_choices.append(output)

        if len(valid_choices) == 0:
            logger.error(
                f"No valid completions after {self.max_retries} attempts. "
                f"With the last response: {response}"
            )
            return index, []

        valid_choices = valid_choices[:n]

        for vc in valid_choices:
            logger.info(f"Valid choice: {vc['output']} - {vc['is_valid']}")
        logger.info(f"Total valid choices: {len(valid_choices)}")

        all_valid = all(vc["is_valid"] for vc in valid_choices)
        if use_cache and self._cache and all_valid:
            self._cache.set(cache_key_data, valid_choices, ttl=cache_ttl)

        return index, valid_choices

    def batch_generate(
        self, batch_messages: List[List[Dict[str, str]]], **kwargs
    ) -> List[List[Dict]]:
        """Batch generate with deduplication and ThreadPoolExecutor."""
        results: List[Optional[List[Dict]]] = [None] * len(batch_messages)

        message_to_indices: Dict[str, List[int]] = {}
        unique_messages: List[List[Dict[str, str]]] = []
        unique_indices: List[int] = []

        for idx, messages in enumerate(batch_messages):
            message_key = json.dumps(messages, sort_keys=True)
            if message_key not in message_to_indices:
                message_to_indices[message_key] = []
                unique_messages.append(messages)
                unique_indices.append(idx)
            message_to_indices[message_key].append(idx)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_index = {
                executor.submit(
                    self.generate, unique_idx, unique_messages[i], **kwargs
                ): unique_idx
                for i, unique_idx in enumerate(unique_indices)
            }
            for future in as_completed(future_to_index):
                try:
                    index, choices = future.result()
                    messages = batch_messages[index]
                    message_key = json.dumps(messages, sort_keys=True)
                    for duplicate_idx in message_to_indices[message_key]:
                        results[duplicate_idx] = choices
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"Error processing batch item {index}: {e}")
                    messages = batch_messages[index]
                    message_key = json.dumps(messages, sort_keys=True)
                    for duplicate_idx in message_to_indices[message_key]:
                        results[duplicate_idx] = []

        return results

    def execute_role(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        **kwargs,
    ) -> Tuple[List[List[Any]], List[Optional[str]]]:
        """Execute a role with optional structured output parsing."""
        output_schema = kwargs.get("output_schema", None)
        schema_keys = None
        schema_value_types = None

        if output_schema is not None:
            assert issubclass(output_schema, pydantic.BaseModel)
            schema_keys = list(output_schema.model_fields.keys())
            schema_value_types = [
                field.annotation.__name__
                for field in output_schema.model_fields.values()
            ]

        if isinstance(messages[0], list):
            results = self.batch_generate(messages, **kwargs)
        else:
            _, results = self.generate(0, messages, **kwargs)
            results = [results]

        all_outputs: List[Optional[List[Any]]] = [None] * len(results)
        all_raw_outputs: List[Optional[str]] = [None] * len(results)

        for idx, res in enumerate(results):
            outputs: List[Any] = []
            raw_output = None

            for item in res:
                if raw_output is None and item["is_valid"]:
                    raw_output = item["raw_output"]

                if output_schema:
                    if item["is_valid"]:
                        parsed_output = output_schema(**item["output"])
                        outputs.append(parsed_output)
                    else:
                        if isinstance(item["output"], str):
                            data = item["output"]
                        elif isinstance(item["output"], dict):
                            data = json.dumps(item["output"])
                        else:
                            data = str(item["output"])
                        extracted = extract_info_from_text(
                            data, schema_keys, schema_value_types
                        )
                        try:
                            parsed_output = output_schema(**extracted)
                            outputs.append(parsed_output)
                        except Exception as e2:
                            logger.error(
                                f"Failed to parse output even after extraction: {e2}"
                            )
                else:
                    outputs.append(item["output"])

            all_outputs[idx] = outputs
            all_raw_outputs[idx] = raw_output

        return all_outputs, all_raw_outputs

    def get_embeddings(
        self, texts: Union[List[str], str], **kwargs
    ) -> Union[List[List[float]], List[float], None]:
        """Generate embeddings with caching support."""
        assert self.is_embedding, (
            "You are trying to use embedding generation on a non-embedding client."
        )

        use_cache = kwargs.get("use_cache", True)
        cache_ttl = kwargs.get("cache_ttl", None)

        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True

        cache_key_data = {
            "model": self.model_name,
            "texts": texts,
            "type": "embedding",
        }

        if use_cache and self._cache:
            cached_result = self._cache.get(cache_key_data)
            if cached_result is not None:
                logger.info(f"Using cached embeddings for {len(texts)} text(s)")
                if single_text:
                    return cached_result[0]
                return cached_result

        try:
            embeddings = self._openai_client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            if single_text:
                result = embeddings.data[0].embedding
                assert result is not None, "Embedding generation failed"
                result_to_cache = [result]
            else:
                result = [item.embedding for item in embeddings.data]
                assert all(r is not None for r in result), (
                    "Embedding generation failed for some texts"
                )
                result_to_cache = result

            if use_cache and self._cache:
                self._cache.set(cache_key_data, result_to_cache, ttl=cache_ttl)

            return result
        except Exception as e:
            logger.error(
                f"Error during embedding generation: {e} for texts: {texts}"
            )
            return None

    def get_reranking_scores(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> Tuple[List[int], List[str], List[float]]:
        """Rerank documents by relevance to query using yes/no logprobs.
        Returns (sorted_indices, labels, scores) so documents[sorted_indices[i]] is the i-th ranked document.
        """
        if isinstance(documents, str):
            documents = [documents]
        if not documents:
            return [], [], []
        

        last_err: Optional[Exception] = None
        body = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        if instruction is not None:
            body["instruct"] = instruction
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._openai_client.post(
                    path="/rerank",
                    cast_to=dict,
                    body=body,
                )
                if not isinstance(response, list):
                    raise ValueError(
                        f"Unexpected /rerank response type: {type(response)}"
                    )

                sorted_indices: List[int] = []
                scores: List[float] = []
                labels: List[str] = []
                for item in response:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    score = item.get("score")
                    if not isinstance(idx, int) or not isinstance(score, (int, float)):
                        continue
                    if idx < 0 or idx >= len(documents):
                        continue
                    sorted_indices.append(idx)
                    score_value = float(score)
                    scores.append(score_value)
                    labels.append("yes" if score_value > 0.5 else "no")

                if len(sorted_indices) != len(documents):
                    remaining = [
                        i for i in range(len(documents)) if i not in set(sorted_indices)
                    ]
                    sorted_indices.extend(remaining)
                    scores.extend([0.0] * len(remaining))
                    labels.extend(["no"] * len(remaining))
                return sorted_indices, labels, scores
            except Exception as e:
                last_err = e
                logger.warning(f"Error during /rerank attempt {attempt}: {e}")
                time.sleep(2 * attempt)

        logger.error(f"Failed reranking after {self.max_retries} attempts: {last_err}")
        fallback_indices = list(range(len(documents)))
        return fallback_indices, ["no"] * len(documents), [0.0] * len(documents)

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int,
        instruction: Optional[str] = None,
    ) -> List[str]:
        """Return top-k documents reranked by relevance to query."""
        if not documents:
            return []
        sorted_indices, _, _ = self.get_reranking_scores(query, documents, instruction)
        return [documents[i] for i in sorted_indices[:top_k]]

    def get_cache_stats(self) -> Dict[str, Any]:
        if self._cache:
            return self._cache.get_stats()
        return {"enabled": False, "message": "Cache not configured"}

    def clear_cache(self, pattern=None) -> int:
        if self._cache:
            return self._cache.clear_all(pattern)
        return 0

    def close(self):
        if hasattr(self, "_openai_client") and self._openai_client:
            if hasattr(self._openai_client, "close"):
                self._openai_client.close()
        if self._cache:
            self._cache.close()
