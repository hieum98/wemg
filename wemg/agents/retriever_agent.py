import logging
import os
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import numpy as np
import faiss
import datasets

from wemg.agents.base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


class RetrieverAgent:
    def __init__(
            self,
            embedder_config: Dict[str, Any],
            corpus_path: Path,
            index_path: Optional[Path] = None,
            embedder_type: str = 'openai',
            ):
        if embedder_type == 'openai':
            self.type = 'openai'
            if 'is_embedding' not in embedder_config:
                embedder_config['is_embedding'] = True
            self.embedder_config = embedder_config
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
        
        model_name = embedder_config.get('model_name', 'unknown_model')
        self.model_name = model_name
        if 'e5' in model_name.lower():
            self.query_instruction_format = "query: {query}"
            self.candidate_instruction_format = "{text}"
        elif 'gte' in model_name.lower():
            self.query_instruction_format = "{query}"
            self.candidate_instruction_format = "{text}"
        elif 'qwen3' in model_name.lower():
            self.query_instruction_format = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
            self.candidate_instruction_format = "{text}"
        else:
            self.query_instruction_format = "{query}"
            self.candidate_instruction_format = "{text}"
        
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.indexed_corpus: datasets.Dataset = self.get_index()
    
    def get_embedder(self) -> Union[BaseLLMAgent, ]:
        if self.type == 'openai':
            return BaseLLMAgent(
                client_type=self.type,
                **self.embedder_config
            )
        else:
            raise ValueError(f"Unsupported embedder type: {self.type}")
    
    def get_index(self):
        corpus = None
        if not self.corpus_path.exists():
            corpus = datasets.load_dataset(str(self.corpus_path), split="train")
            self.corpus_path = Path(f"./retriever_corpora/{self.corpus_path}")
            self.corpus_path.mkdir(parents=True, exist_ok=True)
        elif str(self.corpus_path).endswith(".jsonl"):
            corpus = datasets.load_dataset('json', data_files=str(self.corpus_path), split="train")
            self.corpus_path = self.corpus_path.parent / self.corpus_path.stem
        elif str(self.corpus_path).endswith(".parquet"):
            corpus = datasets.load_dataset('parquet', data_files=str(self.corpus_path), split="train")
            self.corpus_path = self.corpus_path.parent / self.corpus_path.stem
        else:
            corpus = datasets.load_from_disk(str(self.corpus_path))
        assert corpus is not None, "Failed to load corpus."
        if 'contents' not in corpus.column_names:
            raise ValueError("Corpus must have a 'contents' field.")

        if self.index_path and self.index_path.exists():
            logger.info(f"Loading FAISS index from {self.index_path} for {self.model_name}. Please make sure the index is compatible with the embedder.")
            corpus.load_faiss_index('embedding', str(self.index_path))
        else:
            logger.info(f"No index found at {self.index_path}. Rebuilding index for {self.model_name}.")
            embedder = self.get_embedder()
            corpus = self.build_index(
                embedder=embedder,
                corpus=corpus,
                candidate_instruction_format=self.candidate_instruction_format,
                corpus_save_path=self.corpus_path / "indexed_corpus"
            )
            self.corpus_path = self.corpus_path / "indexed_corpus" 
            # Save the index to disk
            if self.index_path:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                self.index_path = self.corpus_path / "indexed_corpus" / "faiss_index.faiss"
            corpus.save_faiss_index('embedding', str(self.index_path))
        return corpus

    @staticmethod
    def build_index(embedder, corpus: datasets.Dataset, candidate_instruction_format: str, corpus_save_path: Optional[Path] = None):
        # check if corpus has 'contents' field
        if 'contents' not in corpus.column_names:
            raise ValueError("Corpus must have a 'contents' field.")
        # check if placeholder {text} is in candidate_instruction_format
        if '{text}' not in candidate_instruction_format:
            raise ValueError("candidate_instruction_format must contain the placeholder '{text}'.")
        # check if embedder has get_embeddings method
        if not hasattr(embedder, 'get_embeddings'):
            raise ValueError("Embedder must have a 'get_embeddings' method.")
        
        def get_embeddings(batch):
            texts = batch['contents']
            embeddings = []
            texts_with_instructions = [candidate_instruction_format.format(text=text) for text in texts]
            try:
                embeddings = embedder.get_embeddings(texts_with_instructions)
                if embeddings:
                    return {'embedding': embeddings}
                else:
                    logger.error("Embeddings returned are empty.")
                    return {'embedding': [[0.0]] * len(texts)} # Need to be [0.0] to mark for regeneration and compatibility with batch processing
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                return {'embedding': [[0.0]] * len(texts)}

        corpus_with_embeddings = corpus.map(get_embeddings, batched=True, batch_size=32, num_proc=os.cpu_count())
        # get the regen corpus
        regen_corpus = corpus_with_embeddings.filter(lambda x: x['embedding'] == [0.0])
        # Filter out records with None embeddings
        corpus_with_embeddings = corpus_with_embeddings.filter(lambda x: x['embedding'] != [0.0])
        if len(regen_corpus) > 0:
            logger.info(f"Regenerating embeddings for {len(regen_corpus)} records.")
            regen_corpus = regen_corpus.map(get_embeddings, batched=True, batch_size=1, num_proc=1)
            regen_corpus = regen_corpus.filter(lambda x: x['embedding'] != [0.0])
            corpus_with_embeddings: datasets.Dataset = datasets.concatenate_datasets([corpus_with_embeddings, regen_corpus])
        
        # Save the corpus with embeddings into disk as checkpoint
        corpus_with_embeddings.save_to_disk(str(corpus_save_path/"with_embeddings"))
        corpus_without_embeddings = corpus_with_embeddings.remove_columns('embedding')
        corpus_without_embeddings.save_to_disk(str(corpus_save_path/"without_embeddings"))

        # Build FAISS index
        corpus_with_embeddings.add_faiss_index(column='embedding', metric_type=faiss.METRIC_INNER_PRODUCT)
        return corpus_with_embeddings

    def retrieve(self, queries: Union[str, List[str]], top_k: int = 5) -> Union[Tuple[List[str], List[float]], Tuple[List[List[str]], List[List[float]]]]:
        is_single_query = False
        if isinstance(queries, str):
            queries = [queries]
            is_single_query = True
        
        embedder = self.get_embedder()
        query_texts_with_instructions = [self.query_instruction_format.format(query=query) for query in queries]
        query_embeddings = embedder.get_embeddings(query_texts_with_instructions)
        # convert to numpy array
        query_embeddings = np.array(query_embeddings).astype('float32')
        scores, retrieved_examples = self.indexed_corpus.get_nearest_examples_batch('embedding', query_embeddings, k=top_k)
        all_contents = []
        for item in retrieved_examples:
            all_contents.append(item['contents'])
        if is_single_query:
            return all_contents[0], scores[0]
        return all_contents, scores
