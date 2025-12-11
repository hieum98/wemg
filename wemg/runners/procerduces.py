import os
from typing import List, Dict, Tuple, Union
import logging
import asyncio

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents import roles
from wemg.agents.tools import wikidata, web_search
from wemg.runners.memory import Memory

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


async def retrieve_from_kb(
        llm_agent: BaseLLMAgent,
        wikidata_entity_retriever: wikidata.WikidataEntityRetrievalTool,
        wikidata_triple_retriever: wikidata.WikidataKHopTriplesRetrievalTool,
        query: str = None,
        entities: List[Union[roles.open_ie.Entity, wikidata.WikidataEntity]] = None,
        top_k: int = 1,
        n_hops: int = 1,
        ) -> Tuple[List[wikidata.WikiTriple], Dict[roles.open_ie.Entity, wikidata.WikidataEntity]]:
    assert query or entities, "Either query or entities must be provided for knowledge base retrieval."

    if query and not entities:
        # NER Extraction
        logger.info("Starting NER extraction from query.")
        ner_role = roles.open_ie.NERRole()
        ner_messages = ner_role.format_messages(input_data=roles.open_ie.NERInput(text=query))
        ner_kwargs = {
            'output_schema': roles.open_ie.NEROutput,
            'n': 1, # only one response
        }
        ner_response = llm_agent.generator_role_execute(ner_messages, **ner_kwargs)
        ner_output: roles.open_ie.NEROutput = ner_role.parse_response(ner_response)
        entities = ner_output.entities
        logger.info(f"Extracted entities: {entities}")
    
    # Retrieve from Wikidata
    not_wikidata_entities = [ent for ent in entities if isinstance(ent, roles.open_ie.Entity)]
    not_wikidata_entities = set(not_wikidata_entities) # deduplicate
    wikidata_entities = [ent for ent in entities if isinstance(ent, wikidata.WikidataEntity)]
    enitity_dict = {} # mapping from open_ie.Entity to wikidata.WikidataEntity
    retrieved_wikidata_entities = []
    if not_wikidata_entities:
        logger.info("Starting Wikidata entity retrieval.")
        retrieval_tasks = []
        for ent in not_wikidata_entities:
            task = wikidata_entity_retriever.ainvoke({
                "query": ent.name,
                "num_entities": top_k
            })
            retrieval_tasks.append(task)
        retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        for item, result in zip(not_wikidata_entities, retrieval_results):
            if isinstance(result, wikidata.WikidataEntity):
                enitity_dict[item] = result
                retrieved_wikidata_entities.append(result)
        logger.info(f"Retrieved Wikidata entities: {retrieved_wikidata_entities}")
    all_wikidata_entities = wikidata_entities + retrieved_wikidata_entities
    # deduplicate entities based on QID
    unique_entities = {ent.qid: ent for ent in all_wikidata_entities if ent.qid}
    unique_entities_list: List[wikidata.WikidataEntity] = list(unique_entities.values())
    logger.info(f"Gatthered {len(unique_entities_list)} unique Wikidata entities for triple retrieval.")

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
                "update_with_details": True,
            }
        )
    logger.info(f"Retrieved {len(all_triples)} triples from Wikidata.")
    return all_triples, enitity_dict



    

    



