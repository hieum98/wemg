"""Wikidata tool for querying entity information from Wikidata knowledge base."""

# Public API - export all classes and constants that other modules use
from wemg.agents.tools.wikidata.models import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
    WikidataPathBetweenEntities,
)
from wemg.agents.tools.wikidata.constants import (
    DEFAULT_PROPERTIES,
    PROPERTY_LABELS,
)
from wemg.agents.tools.wikidata.api_wrapper import CustomWikidataAPIWrapper
from wemg.agents.tools.wikidata.tools import (
    WikidataEntityRetrievalTool,
    WikidataPropertyRetrievalTool,
    WikidataKHopTriplesRetrievalTool,
    WikidataPathFindingTool,
)

__all__ = [
    # Models
    "WikidataEntity",
    "WikidataProperty",
    "WikiTriple",
    "WikidataPathBetweenEntities",
    # Constants
    "DEFAULT_PROPERTIES",
    "PROPERTY_LABELS",
    # API Wrapper
    "CustomWikidataAPIWrapper",
    # Tools
    "WikidataEntityRetrievalTool",
    "WikidataPropertyRetrievalTool",
    "WikidataKHopTriplesRetrievalTool",
    "WikidataPathFindingTool",
]

