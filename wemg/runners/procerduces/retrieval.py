import os
from typing import List, Dict, Tuple, Union
import logging
import asyncio

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents import roles
from wemg.agents.tools import wikidata, web_search
from wemg.runners.procerduces.extraction import extract_entities_from_text

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


async def retrieve_from_web(query: str, retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent], top_k: int = 5) -> List[str]:
    if isinstance(retriever_agent, web_search.WebSearchTool):
        output: web_search.WebSearchOutput = await retriever_agent.ainvoke({"query": query, "top_k": top_k})
        all_results = [f"{item.title}\n{item.snippet}\n{item.full_text}" for item in output.results if output.is_success]
        return all_results
    elif isinstance(retriever_agent, RetrieverAgent):
        all_contents, _ = retriever_agent.retrieve(query, top_k=top_k)
        return all_contents
    else:
        raise ValueError(f"Unsupported retriever agent type: {type(retriever_agent)}")
    

async def retrieve_entities_from_kb(
        llm_agent: BaseLLMAgent,
        wikidata_entity_retriever: wikidata.WikidataEntityRetrievalTool,
        query: str = None,
        entities: List[Union[roles.open_ie.Entity, wikidata.WikidataEntity]] = None,
        top_k: int = 1,
        ) -> Dict[roles.open_ie.Entity, wikidata.WikidataEntity]:
    assert query or entities, "Either query or entities must be provided for knowledge base retrieval."

    if query and not entities:
        # NER Extraction
        entities = extract_entities_from_text(llm_agent, query)
    
    # Retrieve from Wikidata
    not_wikidata_entities = [ent for ent in entities if isinstance(ent, roles.open_ie.Entity)]
    not_wikidata_entities = set(not_wikidata_entities) # deduplicate
    wikidata_entities = [ent for ent in entities if isinstance(ent, wikidata.WikidataEntity)]
    enitity_dict = {} # mapping from open_ie.Entity to wikidata.WikidataEntity
    retrieved_wikidata_entities = []
    if not_wikidata_entities:
        logger.info("Starting Wikidata entity retrieval.")
        retrieval_results = await wikidata_entity_retriever.ainvoke(
            {
                "query": [ent.name for ent in not_wikidata_entities],
                "num_entities": top_k
            }
        )
        retrieval_results = [[ent for ent in result if isinstance(ent, wikidata.WikidataEntity)] for result in retrieval_results] # filter out non-wikidata results
        for item, result in zip(not_wikidata_entities, retrieval_results):
            if result:
                enitity_dict[item] = result[0] # take the top-1 entity
                retrieved_wikidata_entities.extend(result)
        logger.info(f"Retrieved Wikidata {len(retrieved_wikidata_entities)} entities")
    all_wikidata_entities = wikidata_entities + retrieved_wikidata_entities
    # deduplicate entities based on QID
    unique_entities = {ent.qid: ent for ent in all_wikidata_entities if ent.qid}
    unique_entities_list: List[wikidata.WikidataEntity] = list(unique_entities.values())
    logger.info(f"Gathered {len(unique_entities_list)} unique Wikidata entities.")
    return unique_entities, enitity_dict


async def retrieve_from_kb(
        llm_agent: BaseLLMAgent,
        wikidata_entity_retriever: wikidata.WikidataEntityRetrievalTool,
        wikidata_triple_retriever: wikidata.WikidataKHopTriplesRetrievalTool,
        query: Union[str, List[str]] = None,
        entities: List[Union[roles.open_ie.Entity, wikidata.WikidataEntity]] = None,
        top_k: int = 1,
        n_hops: int = 1,
        ) -> Tuple[List[wikidata.WikiTriple], Dict[roles.open_ie.Entity, wikidata.WikidataEntity]]:
    assert query or entities, "Either query or entities must be provided for knowledge base retrieval."
    if isinstance(query, list):
        # check all start by "Q" or "P"
        assert all(isinstance(q, str) and q.startswith("Q") for q in query), "When query is a list, all items must be Wikidata QIDs (start with 'Q')."
        unique_entities_list = [wikidata.WikidataEntity(qid=qid, name="", description="") for qid in query] # Fake entities with only QID to retrieve triples later
        enitity_dict = {}
    else:
        unique_entities_list, enitity_dict = await retrieve_entities_from_kb(
            llm_agent=llm_agent,
            wikidata_entity_retriever=wikidata_entity_retriever,
            query=query,
            entities=entities,
            top_k=top_k,
        )

    # Retrieve k-hop triples
    all_triples = []
    if unique_entities_list:
        logger.info("Starting k-hop triple retrieval from Wikidata.")
        all_qids = [ent.qid for ent in unique_entities_list]
        all_triples = await wikidata_triple_retriever.ainvoke(
            {
                "query": all_qids,
                "is_qids": True,
                "k": n_hops,
                "bidirectional": False,
                "update_with_details": False,
            }
        )
        # Flatten the list of lists
        all_triples = sum(all_triples, [])
        # deduplicate triples
        all_triples = list(set(all_triples))

    logger.info(f"Retrieved {len(all_triples)} triples from Wikidata.")
    return all_triples, enitity_dict


def get_wiki_id(query: Union[List[str], str], id_type: str, top_k: int=1) -> List[str]:
    """Retrieve Wikidata IDs (item: QID or property: PID) for the given queries."""
    wikidata_wrapper = wikidata.CustomWikidataAPIWrapper(top_k_results=top_k)
    if isinstance(query, str):
        query = [query]
    query = list(set(query))  # deduplicate
    results = wikidata_wrapper._get_id(query, id_type=id_type)
    results = [[item for item in x if item] for x in results]  # filter out empty results
    assert len(results) == len(query), "Mismatch in number of results and queries."
    mapped_results = {q: res for q, res in zip(query, results)}
    return mapped_results
        

if __name__ == "__main__":
    import asyncio
    from wemg.agents.tools import wikidata

    # Configure logging
    log_level = os.getenv("LOGGING_LEVEL", "DEBUG")
    logging.basicConfig(level=log_level)

    entities = [
        roles.open_ie.Entity(name="Barack Obama", description="44th President of the United States", is_scalar=False),
        roles.open_ie.Entity(name="Heisenberg", description="Famous physicist known for the uncertainty principle", is_scalar=False),
        wikidata.WikidataEntity(
            qid="Q917",
            name="Albert Einstein",
            description="German-born theoretical physicist",
        )
    ]

    entity_retriever = wikidata.WikidataEntityRetrievalTool()
    triple_retriever = wikidata.WikidataKHopTriplesRetrievalTool()

    results = asyncio.run(retrieve_from_kb(
        llm_agent=None,
        wikidata_entity_retriever=entity_retriever,
        wikidata_triple_retriever=triple_retriever,
        entities=entities,
        top_k=1,
        n_hops=2,
    ))
    

    



    

    



