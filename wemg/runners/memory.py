from typing import List
import networkx as nx


class Memory:
    def __init__(
            self,
            textual_memory: List[str],
            graph_memory: nx.Graph
    ):  
        self.entity_dict = {} # mapping from open_ie.Entity to wikidata.WikidataEntity
        self.textual_memory = textual_memory
        self.graph_memory = graph_memory
    
    def add_textual_memory(self, text: str):
        self.textual_memory.append(text)
    
    def consolidate_textual_memory(self):
        # Placeholder for consolidation logic
        pass

    def remove_textual_memory(self):
        pass
    
    def update_textual_memory(self):
        pass

    def add_graph_node(self):
        pass

    def add_graph_edge(self):
        pass

    def remove_graph_node(self):
        pass

    def remove_graph_edge(self):
        pass

    def update_graph_node(self):
        pass

    def update_graph_edge(self):
        pass

    def consolidate_graph_memory(self):
        pass


    
    

