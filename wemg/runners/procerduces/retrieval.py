import os
from typing import List, Dict, Optional, Tuple, Union
import logging
import asyncio

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent
from wemg.agents import roles
from wemg.agents.tools import wikidata, web_search
from wemg.runners.interaction_memory import InteractionMemory
from wemg.runners.procerduces.base_role_excercution import execute_role

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
        wikidata_property_retriever: wikidata.WikidataPropertyRetrievalTool,
        query: str = None,
        entities: List[Union[roles.open_ie.Entity, wikidata.WikidataEntity]] = [],
        relations: List[Union[str, wikidata.WikidataProperty]] = [],
        top_k_entities: int = 1,
        top_k_properties: int = 1,
        interaction_memory: Optional[InteractionMemory] = None,
        ):
    assert query or entities, "Either query or entities must be provided for knowledge base retrieval."

    if query:
        # Query generation
        graph_query_input = roles.generator.QueryGraphGeneratorInput(input_text=query)
        responses, graph_query_log = await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.StructuredQueryGenerator(),
            input_data=graph_query_input,
            interaction_memory=interaction_memory,
            n=1
        )
        queries: List[roles.generator.Query] = responses[0].queries
        for q in queries:
            subject_entity = roles.open_ie.Entity(name=q.subject)
            entities.append(subject_entity)
            relations.append(q.relation)
    
    # Retrieve from Wikidata
    not_wikidata_properties = []
    wikidata_properties = []
    if relations:
        not_wikidata_properties = [rel for rel in relations if isinstance(rel, str)]
        not_wikidata_properties = list(set(not_wikidata_properties)) # deduplicate
        wikidata_properties = [rel for rel in relations if isinstance(rel, wikidata.WikidataProperty)]
    property_dict = {}
    if not_wikidata_properties:
        logger.info(f"Starting Wikidata property retrieval for properties: {not_wikidata_properties}")
        retrieval_results = await wikidata_property_retriever.ainvoke(
            {
                "query": not_wikidata_properties,
                "top_k_results": top_k_properties
            }
        )
        retrieval_results = [[prop for prop in result if isinstance(prop, wikidata.WikidataProperty)] for result in retrieval_results] # filter out non-wikidata results
        for item, result in zip(not_wikidata_properties, retrieval_results):
            if result:
                property_dict[item] = result
                wikidata_properties.extend(result)
        logger.info(f"Retrieved {len(wikidata_properties)} Wikidata properties.")
    all_properties = list(set(wikidata_properties))

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
                "num_entities": top_k_entities
            }
        )
        retrieval_results = [[ent for ent in result if isinstance(ent, wikidata.WikidataEntity)] for result in retrieval_results] # filter out non-wikidata results
        for item, result in zip(not_wikidata_entities, retrieval_results):
            if result:
                enitity_dict[item] = result
                retrieved_wikidata_entities.extend(result)
        logger.info(f"Retrieved Wikidata {len(retrieved_wikidata_entities)} entities")
    all_wikidata_entities = wikidata_entities + retrieved_wikidata_entities
    # deduplicate entities based on QID
    unique_entities = list(set(all_wikidata_entities))
    logger.info(f"Gathered {len(unique_entities)} unique Wikidata entities.")
    return unique_entities, enitity_dict, all_properties, property_dict, graph_query_log


async def retrieve_triples(
        wikidata_triple_retriever: wikidata.WikidataKHopTriplesRetrievalTool,
        entities: List[wikidata.WikidataEntity],
        n_hops: int = 1,
        ) -> List[wikidata.WikiTriple]:

    # Retrieve k-hop triples
    all_triples = []
    if entities:
        logger.info("Starting k-hop triple retrieval from Wikidata.")
        all_qids = [ent.qid for ent in entities if isinstance(ent, wikidata.WikidataEntity)]
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
    return all_triples
        

async def retrieve_from_kb(
        llm_agent: BaseLLMAgent,
        question: str,
        entities: List[Union[roles.open_ie.Entity, wikidata.WikidataEntity]] = [],
        relations: List[Union[str, wikidata.WikidataProperty]] = [],
        top_k_entities: int = 1,
        top_k_properties: int = 1,
        n_hops: int = 1,
        use_question_for_graph_retrieval: bool = False,
        interaction_memory: Optional[InteractionMemory] = None,
        ):
    # Retrieve Wikidata entities and props
    wikidata_entity_retriever = wikidata.WikidataEntityRetrievalTool()
    wikidata_property_retriever = wikidata.WikidataPropertyRetrievalTool()
    entities, entity_dict, relations, property_dict, graph_query_log = await retrieve_entities_from_kb(
        llm_agent=llm_agent,
        wikidata_entity_retriever=wikidata_entity_retriever,
        wikidata_property_retriever=wikidata_property_retriever,
        query=question if use_question_for_graph_retrieval else None,
        entities=entities,
        relations=relations,
        top_k_entities=top_k_entities,
        top_k_properties=top_k_properties,
        interaction_memory=interaction_memory
    )

    wikidata_wrapper = wikidata.CustomWikidataAPIWrapper(
        wikidata_props=[r.pid for r in relations],
        wikidata_props_with_labels={r.pid: {'label': r.label, 'description': r.description} for r in relations}
    )
    wikidata_triple_retriever = wikidata.WikidataKHopTriplesRetrievalTool(wikidata_wrapper=wikidata_wrapper)

    retrieved_triples = await retrieve_triples(
        wikidata_triple_retriever=wikidata_triple_retriever,
        entities=entities,
        n_hops=n_hops
    )

    return retrieved_triples, entity_dict, property_dict, graph_query_log


async def explore(
        llm_agent: BaseLLMAgent,
        retriever_agent: Union[web_search.WebSearchTool, RetrieverAgent],
        question: str,
        entities: List[Union[roles.open_ie.Entity, wikidata.WikidataEntity]] = [],
        relations: List[Union[str, wikidata.WikidataProperty]] = [],
        top_k_websearch: int = 5,
        top_k_entities: int = 1,
        top_k_properties: int = 1,
        n_hops: int = 1,
        use_question_for_graph_retrieval: bool = False,
        interaction_memory: Optional[InteractionMemory] = None,
        ):
    # webseach
    # queries, web_retriever_to_log_data = await generate_web_queries(llm_agent=llm_agent, question=question)
    websearch_query_input = roles.generator.QueryGeneratorInput(input_text=question)
    responses, websearch_query_log =  await execute_role(
            llm_agent=llm_agent,
            role=roles.generator.QueryGenerator(),
            input_data=websearch_query_input,
            interaction_memory=interaction_memory,
            n=1
        )
    queries: List[str] = responses[0].queries
    # Concurently run the websearch
    tasks = [retrieve_from_web(query, retriever_agent=retriever_agent, top_k=top_k_websearch) for query in queries]
    documents = await asyncio.gather(*tasks)
    documents = sum(documents, [])
    documents = list(set(documents))

    # KB search
    retrieved_triples, entity_dict, property_dict, graph_query_log = await retrieve_from_kb(
        llm_agent=llm_agent,
        question=question,
        entities=entities,
        relations=relations,
        top_k_entities=top_k_entities,
        top_k_properties=top_k_properties,
        n_hops=n_hops,
        use_question_for_graph_retrieval=use_question_for_graph_retrieval,
        interaction_memory=interaction_memory
    )
    # Process log data
    all_log_keys = set(list(websearch_query_log.keys()) + list(graph_query_log.keys()))
    to_log_data = {key: websearch_query_log.get(key, []) + graph_query_log.get(key, []) for key in all_log_keys}
    return documents, retrieved_triples, entity_dict, property_dict, to_log_data


if __name__ == "__main__":
    import asyncio
    from wemg.agents.tools import wikidata

    

    



    

    



