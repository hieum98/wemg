"""Pydantic models for Wikidata entities, properties, and triples."""

from typing import List, Optional, Any

import pydantic


class WikidataEntity(pydantic.BaseModel):
    """Represents a single Wikidata entity with structured information."""
    
    qid: str = pydantic.Field(..., description="Wikidata QID (e.g., Q7251)")
    label: Optional[str] = pydantic.Field("", description="The entity label/name")
    description: Optional[str] = pydantic.Field("", description="Brief description of the entity")
    aliases: Optional[List[str]] = pydantic.Field(default_factory=list, description="Alternative names for the entity")
    url: Optional[str] = pydantic.Field(None, description="Wikidata URL for the entity")
    wikipedia_url: Optional[str] = pydantic.Field(None, description="Wikipedia URL for the entity")
    wikidata_content: Optional[str] = pydantic.Field(None, description="All related data of the entity from Wikidata")
    wikipedia_content: Optional[str] = pydantic.Field(None, description="All related data of the entity from Wikipedia")

    def to_context(self, include_wiki_page: bool = False) -> str:
        """Return an LLM-friendly, natural-language summary of this entity."""
        label = (self.label or "").strip() or self.qid
        description = (self.description or "").strip()
        header = label
        if description:
            header = f"{header}: {description}"
        lines: List[str] = [header]
        if include_wiki_page and self.wikipedia_content:
            lines.append(f"Wikipedia Content:\n{self.wikipedia_content.strip()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        label = (self.label or "").strip() or self.qid
        description = (self.description or "").strip()
        return f"{label} - {description}" if description else label
    
    def __hash__(self):
        return hash(self.qid)
    
    def __eq__(self, other):
        """Compare entities by QID only."""
        if not isinstance(other, WikidataEntity):
            return False
        return self.qid == other.qid


class WikidataProperty(pydantic.BaseModel):
    """Represents a property of a Wikidata entity."""
    
    pid: str = pydantic.Field(..., description="Wikidata Property ID (e.g., P31)")
    label: Optional[str] = pydantic.Field("", description="The property label/name")
    description: Optional[str] = pydantic.Field(None, description="The description associated with the property")

    def __hash__(self):
        return hash(self.pid)
    
    def __eq__(self, other):
        """Compare properties by PID only."""
        if not isinstance(other, WikidataProperty):
            return False
        return self.pid == other.pid
    
    def __str__(self) -> str:
        """Return string representation with fallback to PID if label is missing."""
        label = (self.label or "").strip() or self.pid
        description = (self.description or "").strip()
        return f"{label}: {description}" if description else label


class WikiTriple(pydantic.BaseModel):
    """Represents a single triple (subject, predicate, object) from Wikidata."""
    
    subject: WikidataEntity = pydantic.Field(..., description="The subject entity of the triple")
    relation: WikidataProperty = pydantic.Field(..., description="The property/relation of the triple")
    object: Any = pydantic.Field(..., description="The object/value of the triple")

    def __str__(self) -> str:
        subject_str = str(self.subject)
        relation_str = str(self.relation)
        object_str = str(self.object)
        return f"Subject: {subject_str}\nRelation: {relation_str}\nObject: {object_str}"

    def __hash__(self):
        if isinstance(self.object, WikidataEntity):
            return hash((self.subject.qid, self.relation.pid, self.object.qid))
        else:
            return hash((self.subject.qid, self.relation.pid, str(self.object)))
    
    def __eq__(self, other):
        """Compare triples by (subject, relation, object) tuple."""
        if not isinstance(other, WikiTriple):
            return False
        if isinstance(self.object, WikidataEntity) and isinstance(other.object, WikidataEntity):
            return (self.subject.qid == other.subject.qid and 
                    self.relation.pid == other.relation.pid and 
                    self.object.qid == other.object.qid)
        elif not isinstance(self.object, WikidataEntity) and not isinstance(other.object, WikidataEntity):
            return (self.subject.qid == other.subject.qid and 
                    self.relation.pid == other.relation.pid and 
                    str(self.object) == str(other.object))
        return False


class WikidataPathBetweenEntities(pydantic.BaseModel):
    """Represents a path between two Wikidata entities."""
    
    source: WikidataEntity = pydantic.Field(..., description="The source entity of the path")
    target: WikidataEntity = pydantic.Field(..., description="The target entity of the path")
    path: List[WikiTriple] = pydantic.Field(default_factory=list, description="List of triples forming the path from source to target")
    path_length: int = pydantic.Field(0, description="Length of the path (number of hops)")
    
    def __str__(self) -> str:
        if not self.path:
            return f"No path found between {self.source} and {self.target}."
        all_triples = []
        for i, triple in enumerate(self.path):
            triple_str = f"{i + 1}.\n{str(triple)}"
            all_triples.append(triple_str)
        path_str = "\n--------------\n".join(all_triples)
        return f"Path from {self.source} to {self.target}:\n{path_str}"

    def __hash__(self):
        # Sort triples by their subject.qid, relation.pid and object.qid/str for consistent hashing
        sorted_triples = sorted(
            self.path,
            key=lambda t: (t.subject.qid, t.relation.pid, t.object.qid if isinstance(t.object, WikidataEntity) else str(t.object))
        )
        hashable_path = tuple(
            (triple.subject.qid, triple.relation.pid, triple.object.qid if isinstance(triple.object, WikidataEntity) else str(triple.object))
            for triple in sorted_triples
        )
        return hash(hashable_path)  # This ensures that the hash is order-independent

