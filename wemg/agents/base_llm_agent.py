import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import openai
from openai import ChatCompletion
import pydantic
import litellm

from wemg.utils.parsing import extract_info_from_text
from wemg.utils.caching import RedisCacheManager

litellm.drop_params=True
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class BaseClient:
    def __init__(
            self, 
            model_name: str,
            url: str, 
            api_key: Union[str, None] = None,
            concurrency: int = 64,
            aws_profile_name: Union[str, None] = None,
            max_retries: int = 1,
            cache_config: Optional[Dict[str, Any]] = None,
            is_embedding: bool = False,
            **generate_kwargs: Dict[str, Any],
            ):
        self.model_name = model_name
        self.url = url
        self.api_key = api_key
        self.aws_profile_name = aws_profile_name
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.is_embedding = is_embedding

        ## Generate kwargs
        self.timeout = generate_kwargs.get('timeout', 60)
        self.temperature = generate_kwargs.get('temperature', 0.7)
        self.num_samples = generate_kwargs.get('n', 1)
        self.top_p = generate_kwargs.get('top_p', 0.8)
        self.max_tokens = generate_kwargs.get('max_tokens', 8192) # default max tokens to generate
        self.max_inputs_tokens = generate_kwargs.get('max_input_tokens', 32768) # default max input tokens
        self.top_k = generate_kwargs.get('top_k', 20)
        self.enable_thinking = generate_kwargs.get('enable_thinking', True) # enable chain-of-thought by default
        self.random_seed = generate_kwargs.get('random_seed', None)
        self.structure_output_supported = False
        
        # Initialize Redis cache if config is provided
        self.cache = None
        self.use_cache = False
        if cache_config and cache_config.pop('enabled', False):
            try:
                self.cache = RedisCacheManager(**cache_config)
                self.use_cache = True
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.cache = None

    def prepare_model_kwargs(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[litellm.ModelResponse, ChatCompletion]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate(self, index: int, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        should_return_reasoning = kwargs.get('should_return_reasoning', True)
        output_schema = kwargs.get('output_schema', None)
        use_cache = kwargs.get('use_cache', self.use_cache)
        cache_ttl = kwargs.get('cache_ttl', None)
        
        if output_schema is not None:
            if self.structure_output_supported:
                assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
            else:
                print(f"Model {self.model_name} does not support structured output. Ignoring output schema.")
        
        # Prepare model kwargs for caching and generation
        model_kwargs = self.prepare_model_kwargs(**kwargs)
        n = model_kwargs.get('n', 1)
        
        # Create cache key data
        cache_key_data = {
            'model': self.model_name,
            'messages': messages,
            'kwargs': {k: v for k, v in model_kwargs.items() if k not in ['timeout']},  # Exclude non-deterministic params
        }
        
        # Try to get from cache
        if use_cache and self.cache:
            cached_result = self.cache.get(cache_key_data)
            if cached_result is not None:
                logger.info(f"Using cached result for request {index}")
                return index, cached_result
        
        # Generate if not in cache
        valid_choices = []
        attempts = 0
        while len(valid_choices) < n and attempts < self.max_retries:
            attempts += 1
            try:
                messages = litellm.utils.trim_messages(messages, max_tokens=self.max_inputs_tokens)
                response: litellm.ModelResponse = self.completion(messages, **model_kwargs)
            except Exception as e:
                logger.warning(f"Error during completion: {e}. Retrying...")
                response = None
                time.sleep(2 * attempts)
            if response is None or not hasattr(response, 'choices'):
                # Slightly adjust generation parameters tp shake things up
                model_kwargs['temperature'] = min(1.0, model_kwargs.get('temperature', 0.7) + 0.1 * attempts)
                model_kwargs['top_p'] = min(1.0, model_kwargs.get('top_p', 0.8) + 0.1 * attempts)
                continue
            for choice in response.choices:
                if not choice.message or not choice.message.content:
                    continue
                reasoning = None
                if should_return_reasoning:
                    try:
                        reasoning = choice.message.reasoning_content 
                    except Exception:
                        reasoning = None
                if output_schema and self.structure_output_supported:
                    try:
                        parsed_output = output_schema.model_validate_json(choice.message.content)
                        output = {
                            'output': parsed_output.model_dump() if hasattr(parsed_output, 'model_dump') else parsed_output,
                            'reasoning': reasoning,
                            'is_valid': True
                        }
                    except:
                        logger.warning(f"Failed to parse structured output for {choice.message.content}")
                        if attempts == self.max_retries - 1:
                            output = {
                                'output': choice.message.content,
                                'reasoning': reasoning,
                                'is_valid': False
                            }
                else:
                    output = {
                        'output': choice.message.content,
                        'reasoning': reasoning,
                        'is_valid': True
                    }
                valid_choices.append(output)
        
        if len(valid_choices) == 0:
            logger.error(f"No valid completions after {self.max_retries} attempts. With the last response: {response}")
            return index, []
        
        valid_choices = valid_choices[:n]
        
        # Cache the result which are all valid
        all_valid = all([vc['is_valid'] for vc in valid_choices])
        if use_cache and self.cache and all_valid:
            self.cache.set(cache_key_data, valid_choices, ttl=cache_ttl)
        
        return index, valid_choices

    def batch_generate(
        self, 
        batch_messages: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        # Prepare the results list to maintain order
        results = [None] * len(batch_messages)
        
        # Deduplicate messages to avoid race conditions in caching
        # Map each unique message to its indices in the batch
        message_to_indices = {}
        unique_messages = []
        unique_indices = []
        
        for idx, messages in enumerate(batch_messages):
            # Create a hashable key from messages
            message_key = json.dumps(messages, sort_keys=True)
            if message_key not in message_to_indices:
                message_to_indices[message_key] = []
                unique_messages.append(messages)
                unique_indices.append(idx)
            message_to_indices[message_key].append(idx)
        
        # Use ThreadPoolExecutor for concurrent processing of unique messages
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # Submit tasks only for unique messages
            future_to_index = {
                executor.submit(self.generate, unique_idx, unique_messages[i], **kwargs): unique_idx
                for i, unique_idx in enumerate(unique_indices)
            }
            # Process completed futures
            futures_iter = as_completed(future_to_index)
            for future in futures_iter:
                try:
                    index, choices = future.result()
                    # Get the original messages for this result
                    messages = batch_messages[index]
                    message_key = json.dumps(messages, sort_keys=True)
                    
                    # Replicate the result to all indices with the same message
                    for duplicate_idx in message_to_indices[message_key]:
                        results[duplicate_idx] = choices
                except Exception as e:
                    # Get the index for error logging
                    index = future_to_index[future]
                    logger.error(f"Error processing batch item {index}: {e}")
                    messages = batch_messages[index]
                    message_key = json.dumps(messages, sort_keys=True)
                    # Set error for all duplicates
                    for duplicate_idx in message_to_indices[message_key]:
                        results[duplicate_idx] = []
        return results 
    
    def embedding(self, texts: Union[List[str], str], **kwargs) -> Union[List[List[float]], List[float]]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False, "message": "Cache not configured"}
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys (default: all cache keys)
            
        Returns:
            Number of keys deleted
        """
        if self.cache:
            return self.cache.clear_all(pattern)
        return 0
    
    def close(self):
        """Close client connections including cache."""
        if hasattr(self, 'client') and self.client:
            if hasattr(self.client, 'close'):
                self.client.close()
        if self.cache:
            self.cache.close()


class OpenAIClient(BaseClient):
    def __init__(
            self, 
            model_name: str,
            url: str,
            api_key: Union[str, None] = None,
            concurrency: int = 64,
            aws_profile_name: Union[str, None] = None,
            max_retries: int = 1,
            cache_config: Optional[Dict[str, Any]] = None,
            is_embedding: bool = False, 
            **generate_kwargs: Dict[str, Any]
            ):
        super().__init__(
            model_name=model_name,
            url=url,
            api_key=api_key,
            concurrency=concurrency,
            aws_profile_name=aws_profile_name,
            max_retries=max_retries,
            cache_config=cache_config,
            is_embedding=is_embedding,
            **generate_kwargs
        )
        self.structure_output_supported = True

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.url
        )

    def prepare_model_kwargs(self, **kwargs) -> Dict[str, Any]:
        enable_thinking = kwargs.get('enable_thinking', self.enable_thinking)
        output_schema = kwargs.get('output_schema', None)
        response_format = None
        if output_schema is not None:
            assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": output_schema.model_json_schema()
                    }
            }
        model_kwargs = {
            'model': self.model_name,
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'top_p': kwargs.get('top_p', self.top_p),
            'n': kwargs.get('n', self.num_samples),
            'seed': kwargs.get('random_seed', self.random_seed),
            'response_format': response_format,
            'timeout': kwargs.get('timeout', self.timeout),
            'logprobs': kwargs.get('logprobs', None),
            'top_logprobs': kwargs.get('top_logprobs', None),
            'extra_body': {
                "top_k": kwargs.get('top_k', self.top_k),
                "chat_template_kwargs": {"enable_thinking": True if enable_thinking else None},
            }
        }
        return model_kwargs
    
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    
    def embedding(self, texts: Union[List[str], str], **kwargs) -> Union[List[List[float]], List[float]]:
        assert self.is_embedding == True, "You are trying to use embedding generation on a non-embedding client."
        
        use_cache = kwargs.get('use_cache', True)
        cache_ttl = kwargs.get('cache_ttl', None)
        
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Create cache key data
        cache_key_data = {
            'model': self.model_name,
            'texts': texts,
            'type': 'embedding'
        }
        
        # Try to get from cache
        if use_cache and self.cache:
            cached_result = self.cache.get(cache_key_data)
            if cached_result is not None:
                logger.info(f"Using cached embeddings for {len(texts)} text(s)")
                if single_text:
                    return cached_result[0]
                return cached_result
        
        try:
            # Generate embeddings
            embeddings = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            if single_text:
                result = embeddings.data[0].embedding
                assert result is not None, "Embedding generation failed"
                result_to_cache = [result]
            else:
                result = [item.embedding for item in embeddings.data]
                assert all(r is not None for r in result), "Embedding generation failed for some texts"
                result_to_cache = result
            # Cache the result
            if use_cache and self.cache:
                self.cache.set(cache_key_data, result_to_cache, ttl=cache_ttl)
            return result
        except Exception as e:
            logger.error(f"Error during embedding generation: {e} for texts: {texts}")
            return None


class BaseLLMAgent:
    def __init__(self, client_type: str = 'openai', **client_kwargs):
        """Initialize the BaseLLMAgent with specified client type and configurations.
        Args:
            client_type: Type of the LLM client (default: 'openai')
            **client_kwargs: Additional keyword arguments for client configuration which may include:
                - model_name: Name of the model to use
                - url: API endpoint URL
                - api_key: API key for authentication
                - concurrency: Number of concurrent requests
                - aws_profile_name: AWS profile name if applicable
                - max_retries: Maximum number of retries for failed requests
                - cache_config: Configuration dictionary for caching
                - is_embedding: Boolean indicating if the client is for embedding generation
                - Other model-specific generation parameters
        """
        assert client_type in ['openai'], f"Unsupported client type: {client_type}"
        self.client_kwargs = client_kwargs
        self.client_type = client_type

    def get_client(self) -> BaseClient:
        if self.client_type == 'openai':
            return OpenAIClient(**self.client_kwargs)
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

    def generator_role_execute(self, 
                     messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
                     **kwargs
                     ) -> List[List[Any]]:
        output_schema = kwargs.get('output_schema', None)
        if output_schema is not None:
            assert issubclass(output_schema, pydantic.BaseModel), "Output schema must be a subclass of pydantic.BaseModel"
        client = self.get_client()
        if isinstance(messages[0], list):
            results = client.batch_generate(messages, **kwargs)
        else:
            _, results = client.generate(0, messages, **kwargs)
            results = [results]
        all_outputs = [None] * len(results)
        for idx, res in enumerate(results):
            outputs = []
            for item in res:
                # Convert output to pydantic model if schema is provided and output is valid
                if output_schema:
                    if item['is_valid']:
                        parsed_output = output_schema(**item['output'])
                        outputs.append(parsed_output)
                    else:
                        keys = output_schema.model_fields.keys()
                        value_types = [field.annotation.__name__ for field in output_schema.model_fields.values()]
                        if isinstance(item['output'], str):
                            data = item['output']
                        elif isinstance(item['output'], dict):
                            data = json.dumps(item['output'])
                        else:
                            data = str(item['output'])
                        extracted = extract_info_from_text(data, keys, value_types)
                        try:
                            parsed_output = output_schema(**extracted)
                            outputs.append(parsed_output)
                        except Exception as e2:
                            logger.error(f"Failed to parse output even after extraction: {e2}")
                else:
                    outputs.append(item['output'])
            all_outputs[idx] = outputs
        return all_outputs
    
    def get_embeddings(self, inputs: Union[List[str], str], **kwargs):
        client = self.get_client()
        return client.embedding(inputs, **kwargs)
    
    def _calculate_score(self, logprobs_content) -> Tuple[str, float]:
        """
        Extracts 'yes' and 'no' logprobs and calculates the softmax score.
        """
        # Dictionary to store logprobs for target tokens
        token_probs = {"yes": -9999.0, "no": -9999.0}
        
        # logprobs_content is a list of TopLogprob objects. 
        # We only look at the first generated token.
        if not logprobs_content:
            return "no", 0.0

        first_token_logprobs = logprobs_content[0].top_logprobs
        
        for token_data in first_token_logprobs:
            # Clean the token string (remove whitespace/lower case if needed, though Qwen is usually precise)
            token_str = token_data.token.strip().lower()
            if token_str in token_probs:
                token_probs[token_str] = token_data.logprob

        yes_logprob = token_probs["yes"]
        no_logprob = token_probs["no"]

        # Softmax: score = exp(yes) / (exp(yes) + exp(no))
        # Optimized as sigmoid of difference: 1 / (1 + exp(no - yes))
        try:
            score = 1.0 / (1.0 + np.exp(no_logprob - yes_logprob))
        except OverflowError:
            score = 0.0 if no_logprob > yes_logprob else 1.0

        label = "yes" if score > 0.5 else "no"
        return label, score
    
    def get_reranking_scores(self, query: str, 
                             documents: List[Union[str, List[str]]],
                             instruction: str = None,
                             ) -> Tuple[List[str], List[float]]:
        """Get reranking scores for documents given a query.
        
        Args:
            query: The query string or list of strings.
            documents: List of document strings or list of list of strings.
            **kwargs: Additional keyword arguments for embedding generation.
        Returns:
            List of yes/no labels and their corresponding scores.
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        if isinstance(documents, str):
            documents = [documents]
        def create_messages(query: str, doc: str, instruction: str) -> List[Dict[str, str]]:
            return [
                {
                    "role": "system",
                    "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                },
                {
                    "role": "user",
                    "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
                }
            ]

        client: OpenAIClient = self.get_client()
        all_messages = [create_messages(query, doc, instruction) for doc in documents]
        kwargs = {
            'temperature': 0.0,
            'max_tokens': 1, # only need to generate one token ('yes' or 'no')
            'n': 1,
            'timeout': 30,
            'logprobs': True,
            'top_logprobs': 20,
            'enable_thinking': False,
        }
        with ThreadPoolExecutor(max_workers=client.concurrency) as executor:
            future_to_index = {
                executor.submit(client.completion, messages, **kwargs): idx
                for idx, messages in enumerate(all_messages)
            }
            labels = [None] * len(documents)
            scores = [0.0] * len(documents)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    response: litellm.ModelResponse = future.result()
                    label, score = self._calculate_score(response.choices[0].logprobs.content)
                    labels[idx] = label
                    scores[idx] = score
                except Exception as e:
                    logger.error(f"Error during reranking for document {idx}: {e}")
                    labels[idx] = "no"
                    scores[idx] = 0.0
        return labels, scores




