"""Centralized enrichment logic for Wikidata entities and properties."""

import logging
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

from wemg.agents.tools.wikidata.models import (
    WikidataEntity,
    WikidataProperty,
    WikiTriple,
)

if TYPE_CHECKING:
    from wemg.agents.tools.wikidata.api_wrapper import CustomWikidataAPIWrapper

logger = logging.getLogger(__name__)


class EnrichmentCollector:
    """Collects entities and properties that need enrichment, then enriches them in batch."""
    
    def __init__(self, api_wrapper: "CustomWikidataAPIWrapper"):
        self.api_wrapper = api_wrapper
        self.entity_qids_to_enrich: Set[str] = set()
        self.property_pids_to_enrich: Set[str] = set()
        self.enriched_entities: Dict[str, WikidataEntity] = {}
        self.enriched_properties: Dict[str, WikidataProperty] = {}
    
    def _is_entity_fulfilled(self, entity: WikidataEntity) -> bool:
        """Check if an entity is fully fulfilled with all meaningful data.
        
        An entity is fulfilled only if ALL fields are filled:
        - label (meaningful, not just QID)
        - description
        - aliases (has at least one)
        - url
        - wikipedia_url
        - wikidata_content
        """
        if not entity or not entity.qid:
            return False
        
        # Check if label is meaningful (not just QID)
        has_label = False
        if entity.label:
            label = entity.label.strip()
            has_label = label and not (label.startswith("Q") and label[1:].isdigit())
        if not has_label:
            return False
        
        # Check all other required fields
        if not (entity.description and entity.description.strip()):
            return False
        # if not (entity.aliases and len(entity.aliases) > 0):
        #     return False
        # if not (entity.url and entity.url.strip()):
        #     return False
        # if not (entity.wikipedia_url and entity.wikipedia_url.strip()):
        #     return False
        # if not (entity.wikidata_content and entity.wikidata_content.strip()):
        #     return False
        
        return True
    
    def _is_property_fulfilled(self, prop: WikidataProperty) -> bool:
        """Check if a property is fully fulfilled with all meaningful data.
        
        A property is fulfilled only if ALL fields are filled:
        - label (meaningful, not just PID)
        - description
        """
        if not prop or not prop.pid:
            return False
        
        # Check if label is meaningful (not just PID)
        has_label = False
        if prop.label:
            label = prop.label.strip()
            has_label = label and not (label.startswith("P") and label[1:].isdigit())
        if not has_label:
            return False
        
        # Check description
        if not (prop.description and prop.description.strip()):
            return False
        
        return True
    
    def add_entity(self, entity: WikidataEntity) -> None:
        """Add an entity - store if fulfilled, mark for enrichment otherwise."""
        if not entity or not entity.qid:
            return
        qid = entity.qid.upper()
        if qid in self.enriched_entities:
            return
        if self._is_entity_fulfilled(entity):
            # Entity already has label, store directly
            self.enriched_entities[qid] = entity
        else:
            # Mark for enrichment
            self.entity_qids_to_enrich.add(qid)
    
    def add_entity_qid(self, qid: str) -> None:
        """Mark an entity QID for enrichment (legacy method for QID-only)."""
        if qid:
            qid_upper = qid.upper()
            if qid_upper not in self.enriched_entities:
                self.entity_qids_to_enrich.add(qid_upper)
    
    def add_property(self, prop: WikidataProperty) -> None:
        """Add a property - store if fulfilled, mark for enrichment otherwise."""
        if not prop or not prop.pid:
            return
        pid = prop.pid.upper()
        if pid in self.enriched_properties:
            return
        if self._is_property_fulfilled(prop):
            # Property already has label, store directly
            self.enriched_properties[pid] = prop
            # Also update cache
            self.api_wrapper.wikidata_props_with_labels[pid] = {
                "label": prop.label,
                "description": prop.description
            }
        elif pid not in self.api_wrapper.wikidata_props_with_labels:
            # Mark for enrichment if not in cache
            self.property_pids_to_enrich.add(pid)
    
    def add_property_pid(self, pid: str) -> None:
        """Mark a property PID for enrichment (legacy method for PID-only)."""
        if pid and pid not in self.enriched_properties:
            pid = pid.upper()
            # Check if already in cache
            if pid not in self.api_wrapper.wikidata_props_with_labels:
                self.property_pids_to_enrich.add(pid)
    
    def collect_from_triples(self, triples: List[WikiTriple]) -> None:
        """Collect all entity QIDs and property PIDs from a list of triples.
        
        If entities/properties already have labels, store them directly.
        Otherwise, mark them for enrichment.
        """
        for triple in triples:
            # Collect subject entity
            if triple.subject:
                self.add_entity(triple.subject)
            
            # Collect object entity if it's a WikidataEntity
            if isinstance(triple.object, WikidataEntity):
                self.add_entity(triple.object)
            
            # Collect property
            if triple.relation:
                self.add_property(triple.relation)
    
    def collect_from_entities(self, entities: List[WikidataEntity]) -> None:
        """Collect entity QIDs from a list of entities."""
        for entity in entities:
            if entity:
                self.add_entity(entity)
    
    def enrich_all(self, get_details: bool = False) -> None:
        """Perform batch enrichment of all collected entities and properties.
        
        Only enriches items that are not already fulfilled (have labels).
        """
        # Remove already-enriched items from the to-enrich sets
        self.entity_qids_to_enrich -= set(self.enriched_entities.keys())
        self.property_pids_to_enrich -= set(self.enriched_properties.keys())
        
        # Log stats
        logger.debug(
            f"Enrichment stats: {len(self.enriched_entities)} entities already fulfilled, "
            f"{len(self.entity_qids_to_enrich)} need enrichment; "
            f"{len(self.enriched_properties)} properties already fulfilled, "
            f"{len(self.property_pids_to_enrich)} need enrichment"
        )
        
        # Enrich properties first (they're needed for entity content)
        if self.property_pids_to_enrich:
            logger.info(f"Enriching {len(self.property_pids_to_enrich)} properties")
            properties = self.api_wrapper._get_property(list(self.property_pids_to_enrich))
            if isinstance(properties, list):
                for prop in properties:
                    if prop and prop.pid:
                        pid_upper = prop.pid.upper()
                        self.enriched_properties[pid_upper] = prop
                        # Update cache
                        self.api_wrapper.wikidata_props_with_labels[pid_upper] = {
                            "label": prop.label,
                            "description": prop.description
                        }
            elif properties and properties.pid:
                pid_upper = properties.pid.upper()
                self.enriched_properties[pid_upper] = properties
                self.api_wrapper.wikidata_props_with_labels[pid_upper] = {
                    "label": properties.label,
                    "description": properties.description
                }
        
        # Enrich entities
        if self.entity_qids_to_enrich:
            logger.info(f"Enriching {len(self.entity_qids_to_enrich)} entities")
            entities = self.api_wrapper._get_item(list(self.entity_qids_to_enrich), get_details=get_details)
            if isinstance(entities, list):
                for entity in entities:
                    if entity and entity.qid:
                        self.enriched_entities[entity.qid.upper()] = entity
            elif entities and entities.qid:
                self.enriched_entities[entities.qid.upper()] = entities
    
    def get_enriched_entity(self, qid: str) -> WikidataEntity:
        """Get enriched entity, or return original if not enriched."""
        return self.enriched_entities.get(qid.upper() if qid else None)
    
    def get_enriched_property(self, pid: str) -> WikidataProperty:
        """Get enriched property, or return original if not enriched."""
        return self.enriched_properties.get(pid.upper() if pid else None)
    
    def enrich_triples(self, triples: List[WikiTriple]) -> List[WikiTriple]:
        """Enrich all entities and properties in triples using enriched data.
        
        Uses enriched data if available, otherwise keeps original if already fulfilled.
        """
        enriched_triples: List[WikiTriple] = []
        
        for triple in triples:
            # Get enriched subject or use original if already fulfilled
            subject = self.get_enriched_entity(triple.subject.qid)
            if not subject:
                # Check if original is already fulfilled
                if self._is_entity_fulfilled(triple.subject):
                    subject = triple.subject
                else:
                    # Skip if subject not found and not fulfilled
                    continue
            
            # Get enriched relation or use original if already fulfilled
            relation = triple.relation
            enriched_prop = self.get_enriched_property(relation.pid)
            if enriched_prop:
                relation = enriched_prop
            elif not self._is_property_fulfilled(relation):
                # Update from cache if available
                pid_upper = relation.pid.upper() if relation.pid else None
                if pid_upper and pid_upper in self.api_wrapper.wikidata_props_with_labels:
                    prop_info = self.api_wrapper.wikidata_props_with_labels[pid_upper]
                    relation = WikidataProperty(
                        pid=relation.pid,
                        label=prop_info.get("label", relation.label or ""),
                        description=prop_info.get("description", relation.description)
                    )
            
            # Get enriched object or use original if already fulfilled
            if isinstance(triple.object, WikidataEntity):
                object_entity = self.get_enriched_entity(triple.object.qid)
                if not object_entity:
                    # Check if original is already fulfilled
                    if self._is_entity_fulfilled(triple.object):
                        object_entity = triple.object
                    else:
                        # Skip if object entity not found and not fulfilled
                        continue
                enriched_triples.append(WikiTriple(
                    subject=subject,
                    relation=relation,
                    object=object_entity
                ))
            else:
                # Object is a literal value, keep as is
                enriched_triples.append(WikiTriple(
                    subject=subject,
                    relation=relation,
                    object=triple.object
                ))
        
        return enriched_triples

