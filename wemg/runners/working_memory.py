from enum import Enum
import logging
import os
from typing import Any, Dict, List, Union
import networkx as nx

from wemg.agents import roles
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools import wikidata
from wemg.utils.preprocessing import get_node_id

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))

class Memory:
    def __init__(
            self,
            textual_memory: List[str] = [],
            graph_memory: Union[nx.DiGraph, None] = None,
            max_textual_memory_tokens: int = 8192,
    ):  
        self.entity_dict: Dict[roles.open_ie.Entity, wikidata.WikidataEntity] = {} # mapping from open_ie.Entity to wikidata.WikidataEntity
        self.textual_memory: List[str] = textual_memory
        if graph_memory is None:
            self.graph_memory: nx.DiGraph = nx.DiGraph()
        else:
            self.graph_memory: nx.DiGraph = graph_memory

        self.max_textual_memory_tokens = max_textual_memory_tokens
    
    def add_textual_memory(self, text: str, source: roles.extractor.SourceType=roles.extractor.SourceType.SYSTEM_PREDICTION):
        """Add text to textual memory if not already present."""
        text = text.strip()
        if source == roles.extractor.SourceType.SYSTEM_PREDICTION:
            text = f"[System Prediction]: {text}"
        elif source == roles.extractor.SourceType.RETRIEVAL:
            text = f"[Retrieval]: {text}"
        if text not in self.textual_memory:
            self.textual_memory.append(text)

    def format_textual_memory(self) -> str:
        """Format the textual memory as a single string."""
        memory = [f"- {text.strip()}" for text in self.textual_memory]
        return "\n".join(memory)

    def consolidate_textual_memory(
            self, 
            llm_agent: BaseLLMAgent, 
            question: str, 
            in_context_examples: List[Dict[str, str]] = None,
            ):
        """Consolidate the textual memory with respect to the question."""
        memory_consolidation_role = roles.extractor.MemoryConsolidationRole()
        memory_consolidation_input = roles.extractor.MemoryConsolidationInput(
            question=question,
            memory=self.format_textual_memory()
        )
        memory_consolidation_messages = memory_consolidation_role.format_messages(input_data=memory_consolidation_input, history=in_context_examples)
        consolidation_kwargs = {
            'output_schema': roles.extractor.MemoryConsolidationOutput,
            'n': 1,
            'max_tokens': self.max_textual_memory_tokens
        }
        consolidation_response = llm_agent.generator_role_execute(memory_consolidation_messages, **consolidation_kwargs)[0][0]
        consolidation_output: roles.extractor.MemoryConsolidationOutput = memory_consolidation_role.parse_response(consolidation_response)
        for item in consolidation_output.consolidated_memory:
            if item.provenance not in [roles.extractor.SourceType.SYSTEM_PREDICTION.value, roles.extractor.SourceType.RETRIEVAL.value]:
                self.add_textual_memory(item.content, source=roles.extractor.SourceType.SYSTEM_PREDICTION)
            else:
                self.add_textual_memory(item.content, source=item.provenance)
    
    def consolidate_textual_memory_rerank(self):
        pass

    def remove_textual_memory(self):
        pass
    
    def update_textual_memory(self):
        pass

    def add_edge_to_graph_memory(self, wiki_triple: wikidata.WikiTriple):
        subject_id = get_node_id(wiki_triple.subject)
        object_id = get_node_id(wiki_triple.object)
        if not self.graph_memory.has_node(subject_id):
            self.graph_memory.add_node(subject_id, data=wiki_triple.subject)
        if not self.graph_memory.has_node(object_id):
            self.graph_memory.add_node(object_id, data=wiki_triple.object)
        if not self.graph_memory.has_edge(subject_id, object_id):
            self.graph_memory.add_edge(
                subject_id,
                object_id,
                relation={wiki_triple.relation}
                )
        else:
            # append relation to the existing set
            self.graph_memory.edges[subject_id, object_id]['relation'].add(wiki_triple.relation)

    def synchronize_memories(self):
        pass

    
    

