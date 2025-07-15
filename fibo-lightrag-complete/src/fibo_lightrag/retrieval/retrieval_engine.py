"""
Dual-level retrieval engine combining vector similarity and graph traversal.
This is the core component that implements the LightRAG-style retrieval.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    content: str
    score: float
    source_type: str  # 'vector', 'graph', 'hybrid'
    metadata: Dict[str, Any]
    entities: List[str]
    relationships: List[str]

class FiboRetrievalEngine:
    """Dual-level retrieval engine for financial documents."""
    
    def __init__(self, vector_store, graph_builder, entity_extractor, query_processor):
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        self.entity_extractor = entity_extractor
        self.query_processor = query_processor
        self.knowledge_graph = graph_builder.get_graph()
        
        # Retrieval configuration
        self.config = {
            'vector_weight': 0.6,
            'graph_weight': 0.4,
            'max_results': 10,
            'min_similarity': 0.001,  # Lowered threshold for better recall
            'confidence_threshold': 0.001,  # Lowered for debugging
            'max_graph_depth': 2,
            'enable_relationship_inference': True
        }
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add document to both vector store and knowledge graph."""
        try:
            # Add to vector store
            vector_success = self.vector_store.add_document(doc_id, content, metadata)
            
            # Process for knowledge graph
            graph_success = self._add_to_knowledge_graph(doc_id, content)
            
            logger.info(f"Added document {doc_id}: vector={vector_success}, graph={graph_success}")
            return vector_success and graph_success
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def _add_to_knowledge_graph(self, doc_id: str, content: str) -> bool:
        """Add document content to knowledge graph."""
        try:
            # Extract entities from document
            entities = self.entity_extractor.extract_entities(content)
            
            if not entities:
                logger.warning(f"No entities extracted from document {doc_id}")
                return True  # Not an error, just no entities found
            
            # Add entities as nodes to graph
            nodes_added = self._add_entities_to_graph(entities, doc_id)
            
            # Create chunks for relationship inference
            chunks = self._create_chunks_for_graph(content, doc_id)
            
            # Add chunks to graph and infer relationships
            relationships_added = self._add_chunks_to_graph(chunks, entities, content, doc_id)
            
            logger.info(f"Knowledge graph update for {doc_id}: {nodes_added} nodes, {relationships_added} relationships")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to knowledge graph: {e}")
            return False
    
    def _add_entities_to_graph(self, entities: List, doc_id: str) -> int:
        """Add extracted entities as nodes to the knowledge graph."""
        nodes_added = 0
        
        for entity in entities:
            try:
                success = self.graph_builder.add_entity_to_graph(entity, doc_id)
                if success:
                    nodes_added += 1
            except Exception as e:
                logger.error(f"Failed to add entity {entity.text} to graph: {e}")
        
        return nodes_added
    
    def _create_chunks_for_graph(self, content: str, doc_id: str) -> List[Dict]:
        """Create chunks for relationship inference."""
        chunk_size = 500
        overlap = 100
        chunks = []
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_text = content[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    'id': f"{doc_id}_chunk_{len(chunks)}",
                    'text': chunk_text,
                    'start_pos': i,
                    'end_pos': i + len(chunk_text)
                })
        
        return chunks
    
    def _add_chunks_to_graph(self, chunks: List[Dict], all_entities: List, full_text: str, doc_id: str) -> int:
        """Add chunks to graph and infer relationships between entities."""
        relationships_added = 0
        
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_start = chunk['start_pos']
            chunk_end = chunk['end_pos']
            
            # Find entities that appear in this chunk
            chunk_entities = []
            for entity in all_entities:
                # Check if entity appears in this chunk
                if (entity.start_pos >= chunk_start and entity.end_pos <= chunk_end) or \
                   (entity.start_pos < chunk_end and entity.end_pos > chunk_start):
                    chunk_entities.append(entity)
            
            # Create relationships between entities in the same chunk
            if len(chunk_entities) > 1:
                relationships_added += self._create_intra_chunk_relationships(
                    chunk_entities, chunk_text, doc_id
                )
        
        return relationships_added
    
    def _create_intra_chunk_relationships(self, entities: List, chunk_text: str, doc_id: str) -> int:
        """Create relationships between entities that co-occur in the same chunk."""
        relationships_added = 0
        chunk_lower = chunk_text.lower()
        
        # Relationship keywords and their corresponding relationship types
        relationship_indicators = {
            'partnership': ['partnership', 'partners with', 'joint venture', 'alliance'],
            'owns': ['owns', 'acquired', 'subsidiary', 'parent company', 'controlling interest'],
            'provides_service': ['provides', 'offers', 'delivers', 'services', 'serves'],
            'issues_product': ['issues', 'creates', 'underwrites', 'launches', 'produces'],
            'regulated_by': ['regulated by', 'supervised by', 'oversight', 'compliance'],
            'competes_with': ['competes', 'rival', 'competitor', 'competitive'],
            'collaborates_with': ['collaborates', 'works with', 'cooperation', 'joint'],
            'invests_in': ['invests', 'investment', 'stake', 'shares', 'equity'],
            'co_occurs_with': []  # Default relationship for entities that just appear together
        }
        
        # Create relationships between all pairs of entities in the chunk
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Skip if same entity
                if entity1.text.lower() == entity2.text.lower():
                    continue
                
                # Determine relationship type based on context
                relationship_type = 'co_occurs_with'  # Default
                
                for rel_type, keywords in relationship_indicators.items():
                    if rel_type == 'co_occurs_with':
                        continue
                    
                    for keyword in keywords:
                        if keyword in chunk_lower:
                            relationship_type = rel_type
                            break
                    
                    if relationship_type != 'co_occurs_with':
                        break
                
                # Create the relationship edge
                try:
                    edge = self.graph_builder._create_edge(
                        entity1, entity2, relationship_type, chunk_text, doc_id
                    )
                    
                    if self.knowledge_graph.add_edge(edge):
                        relationships_added += 1
                        logger.debug(f"Added relationship: {entity1.text} -{relationship_type}-> {entity2.text}")
                
                except Exception as e:
                    logger.error(f"Failed to create relationship between {entity1.text} and {entity2.text}: {e}")
        
        return relationships_added
    
    def retrieve(self, query: str, method: str = 'hybrid') -> List[RetrievalResult]:
        """Retrieve relevant information using specified method."""
        try:
            if method == 'vector':
                return self._vector_retrieval(query)
            elif method == 'graph':
                return self._graph_retrieval(query)
            elif method == 'hybrid':
                return self._hybrid_retrieval(query)
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []
    
    def _vector_retrieval(self, query: str) -> List[RetrievalResult]:
        """Retrieve using vector similarity search."""
        results = []
        
        # Search similar documents
        similar_docs = self.vector_store.search_similar(
            query, 
            top_k=self.config['max_results'],
            min_similarity=self.config['min_similarity']
        )
        
        for doc_id, similarity in similar_docs:
            doc = self.vector_store.get_document(doc_id)
            if doc and similarity >= self.config['confidence_threshold']:
                result = RetrievalResult(
                    content=doc.content,
                    score=similarity,
                    source_type='vector',
                    metadata=doc.metadata,
                    entities=[],
                    relationships=[]
                )
                results.append(result)
        
        logger.info(f"Vector retrieval found {len(results)} results for query: {query}")
        return results
    
    def _graph_retrieval(self, query: str) -> List[RetrievalResult]:
        """Retrieve using knowledge graph traversal."""
        results = []
        
        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query)
        
        if not query_entities:
            logger.warning(f"No entities found in query: {query}")
            return results
        
        # Find relevant graph nodes
        relevant_nodes = set()
        for entity in query_entities:
            # Find nodes with similar labels
            for node in self.knowledge_graph.nodes.values():
                if entity.text.lower() in node.label.lower() or node.label.lower() in entity.text.lower():
                    relevant_nodes.add(node.id)
        
        # Expand search using graph traversal
        expanded_nodes = self._expand_graph_search(list(relevant_nodes))
        
        # Convert nodes to retrieval results
        for node_id in expanded_nodes:
            node = self.knowledge_graph.get_node(node_id)
            if node:
                # Get relationships for this node
                edges = self.knowledge_graph.get_edges_for_node(node_id)
                relationships = [f"{edge.relationship}" for edge in edges]
                
                # Calculate relevance score
                score = self._calculate_graph_relevance(node, query_entities)
                
                if score >= self.config['confidence_threshold']:
                    result = RetrievalResult(
                        content=f"{node.label}: {node.properties.get('context', '')}",
                        score=score,
                        source_type='graph',
                        metadata=node.properties,
                        entities=[node.label],
                        relationships=relationships
                    )
                    results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:self.config['max_results']]
        
        logger.info(f"Graph retrieval found {len(results)} results for query: {query}")
        return results
    
    def _expand_graph_search(self, seed_nodes: List[str]) -> Set[str]:
        """Expand search using graph traversal."""
        visited = set(seed_nodes)
        queue = seed_nodes.copy()
        max_depth = self.config['max_graph_depth']
        
        for depth in range(max_depth):
            if not queue:
                break
            
            next_queue = []
            for node_id in queue:
                neighbors = self.knowledge_graph.get_neighbors(node_id)
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        next_queue.append(neighbor.id)
            
            queue = next_queue
        
        return visited
    
    def _calculate_graph_relevance(self, node, query_entities) -> float:
        """Calculate relevance score for a graph node."""
        base_score = 0.5
        
        # Boost score if node label matches query entities
        for entity in query_entities:
            if entity.text.lower() in node.label.lower():
                base_score += 0.3
                break
        
        # Boost score based on node confidence
        node_confidence = node.properties.get('confidence', 0.5)
        base_score += node_confidence * 0.2
        
        # Boost score based on number of relationships
        edges = self.knowledge_graph.get_edges_for_node(node.id)
        relationship_boost = min(0.2, len(edges) * 0.05)
        base_score += relationship_boost
        
        return min(1.0, base_score)
    
    def _hybrid_retrieval(self, query: str) -> List[RetrievalResult]:
        """Combine vector and graph retrieval results."""
        # Get results from both methods
        vector_results = self._vector_retrieval(query)
        graph_results = self._graph_retrieval(query)
        
        # Combine and reweight scores
        all_results = []
        
        for result in vector_results:
            result.score *= self.config['vector_weight']
            all_results.append(result)
        
        for result in graph_results:
            result.score *= self.config['graph_weight']
            result.source_type = 'hybrid'
            all_results.append(result)
        
        # Remove duplicates and sort
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        final_results = unique_results[:self.config['max_results']]
        
        logger.info(f"Hybrid retrieval combined {len(vector_results)} vector + {len(graph_results)} graph = {len(final_results)} final results")
        return final_results
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results, keeping the highest scored ones."""
        seen_content = {}
        unique_results = []
        
        for result in results:
            content_key = result.content[:100].lower()  # Use first 100 chars as key
            
            if content_key not in seen_content:
                seen_content[content_key] = result
                unique_results.append(result)
            else:
                # Keep the result with higher score
                if result.score > seen_content[content_key].score:
                    # Replace the existing result
                    for i, existing in enumerate(unique_results):
                        if existing.content[:100].lower() == content_key:
                            unique_results[i] = result
                            seen_content[content_key] = result
                            break
        
        return unique_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        vector_stats = self.vector_store.get_statistics()
        graph_stats = self.knowledge_graph.get_stats()
        
        return {
            'vector_store': vector_stats,
            'knowledge_graph': graph_stats,
            'configuration': self.config
        }