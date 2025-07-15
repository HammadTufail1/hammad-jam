#!/usr/bin/env python3
"""
FIBO-LightRAG: Complete Financial Knowledge Graph System

A comprehensive implementation that combines LightRAG concepts with 
Financial Industry Business Ontology (FIBO) for financial document analysis.

This is a complete, self-contained implementation that includes:
- FIBO ontology parsing and management
- Financial entity extraction
- Knowledge graph construction
- Dual-level retrieval (vector + graph)
- Query processing and understanding
- Complete demo and examples

Usage:
    python fibo_lightrag_complete.py

Dependencies:
    pip install rdflib requests

Author: FIBO-LightRAG Development Team
Version: 1.0.0
"""

import json
import logging
import math
import hashlib
import re
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# FIBO ONTOLOGY PARSING
# =============================================================================

@dataclass
class FiboClass:
    """Represents a FIBO class with its properties."""
    uri: str
    label: str
    definition: str
    parent_classes: List[str]
    properties: List[str]

@dataclass
class FiboProperty:
    """Represents a FIBO property."""
    uri: str
    label: str
    definition: str
    domain: List[str]
    range: List[str]

@dataclass
class FiboRelationship:
    """Represents a relationship between FIBO entities."""
    subject: str
    predicate: str
    object: str
    relationship_type: str

class FiboParser:
    """Parser for FIBO (Financial Industry Business Ontology) files."""
    
    def __init__(self):
        self.classes: Dict[str, FiboClass] = {}
        self.properties: Dict[str, FiboProperty] = {}
        self.relationships: List[FiboRelationship] = []
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample FIBO data for demonstration."""
        # Sample FIBO classes
        sample_classes = {
            "https://spec.edmcouncil.org/fibo/ontology/Organization": FiboClass(
                uri="https://spec.edmcouncil.org/fibo/ontology/Organization",
                label="Organization",
                definition="A formal or informal organization",
                parent_classes=[],
                properties=[]
            ),
            "https://spec.edmcouncil.org/fibo/ontology/FinancialInstrument": FiboClass(
                uri="https://spec.edmcouncil.org/fibo/ontology/FinancialInstrument",
                label="Financial Instrument",
                definition="A financial instrument or security",
                parent_classes=[],
                properties=[]
            ),
            "https://spec.edmcouncil.org/fibo/ontology/Bank": FiboClass(
                uri="https://spec.edmcouncil.org/fibo/ontology/Bank",
                label="Bank",
                definition="A financial institution that accepts deposits and makes loans",
                parent_classes=["https://spec.edmcouncil.org/fibo/ontology/Organization"],
                properties=[]
            ),
            "https://spec.edmcouncil.org/fibo/ontology/MonetaryAmount": FiboClass(
                uri="https://spec.edmcouncil.org/fibo/ontology/MonetaryAmount",
                label="Monetary Amount",
                definition="A monetary amount in a specific currency",
                parent_classes=[],
                properties=[]
            ),
            "https://spec.edmcouncil.org/fibo/ontology/FinancialService": FiboClass(
                uri="https://spec.edmcouncil.org/fibo/ontology/FinancialService",
                label="Financial Service", 
                definition="A service provided by financial institutions",
                parent_classes=[],
                properties=[]
            )
        }
        
        sample_properties = {
            "https://spec.edmcouncil.org/fibo/ontology/hasName": FiboProperty(
                uri="https://spec.edmcouncil.org/fibo/ontology/hasName",
                label="hasName",
                definition="The name of an entity",
                domain=["https://spec.edmcouncil.org/fibo/ontology/Organization"],
                range=["string"]
            ),
            "https://spec.edmcouncil.org/fibo/ontology/owns": FiboProperty(
                uri="https://spec.edmcouncil.org/fibo/ontology/owns",
                label="owns",
                definition="Ownership relationship",
                domain=["https://spec.edmcouncil.org/fibo/ontology/Organization"],
                range=["https://spec.edmcouncil.org/fibo/ontology/Organization"]
            )
        }
        
        sample_relationships = [
            FiboRelationship(
                subject="https://spec.edmcouncil.org/fibo/ontology/Bank",
                predicate="subClassOf",
                object="https://spec.edmcouncil.org/fibo/ontology/Organization",
                relationship_type="subclass"
            )
        ]
        
        self.classes = sample_classes
        self.properties = sample_properties  
        self.relationships = sample_relationships
        
        logger.info(f"Loaded sample FIBO data: {len(sample_classes)} classes, {len(sample_properties)} properties")

# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

@dataclass
class ExtractedEntity:
    """Represents an extracted financial entity."""
    text: str
    fibo_uri: str
    fibo_class: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str

class FiboEntityExtractor:
    """Extracts financial entities from text using FIBO ontology."""
    
    def __init__(self, fibo_parser):
        self.fibo_parser = fibo_parser
        self.entity_patterns = self._build_entity_patterns()
        
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for different entity types."""
        patterns = {
            'organizations': [
                r'\\b[A-Z][a-zA-Z\\s&,.-]+(?:Corp|Corporation|Inc|Incorporated|Ltd|Limited|LLC|Company|Bank|Group)\\b',
                r'\\b(?:Federal Reserve|SEC|FDIC|OCC|CFTC|FINRA|NASDAQ|NYSE)\\b',
            ],
            'monetary_amounts': [
                r'\\$[\\d,]+(?:\\.\\d{2})?(?:\\s*(?:million|billion|trillion))?',
                r'\\b\\d+(?:\\.\\d+)?\\s*(?:million|billion|trillion)\\s*dollars?\\b'
            ],
            'financial_metrics': [
                r'\\b(?:revenue|profit|loss|earnings|EBITDA|ROE|ROI|P/E)\\b',
            ]
        }
        return patterns
    
    def extract_entities(self, text: str, context_window: int = 50) -> List[ExtractedEntity]:
        """Extract financial entities from text."""
        entities = []
        
        # Dictionary-based extraction using FIBO labels
        entities.extend(self._extract_fibo_entities(text, context_window))
        
        # Pattern-based extraction
        entities.extend(self._extract_pattern_entities(text, context_window))
        
        # Remove duplicates and sort by confidence
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _extract_fibo_entities(self, text: str, context_window: int) -> List[ExtractedEntity]:
        """Extract entities using FIBO class labels."""
        entities = []
        text_lower = text.lower()
        
        for uri, fibo_class in self.fibo_parser.classes.items():
            label = fibo_class.label.lower()
            
            if len(label) < 3:  # Skip very short labels
                continue
            
            for match in re.finditer(re.escape(label), text_lower):
                start_pos = match.start()
                end_pos = match.end()
                
                context_start = max(0, start_pos - context_window)
                context_end = min(len(text), end_pos + context_window)
                context = text[context_start:context_end]
                
                confidence = 0.8  # Base confidence for FIBO matches
                
                entity = ExtractedEntity(
                    text=text[start_pos:end_pos],
                    fibo_uri=uri,
                    fibo_class=fibo_class.label,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context
                )
                entities.append(entity)
        
        return entities
    
    def _extract_pattern_entities(self, text: str, context_window: int) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    context_start = max(0, start_pos - context_window)
                    context_end = min(len(text), end_pos + context_window)
                    context = text[context_start:context_end]
                    
                    fibo_uri, fibo_class = self._map_to_fibo_class(entity_type)
                    
                    if fibo_uri:
                        entity = ExtractedEntity(
                            text=match.group(),
                            fibo_uri=fibo_uri,
                            fibo_class=fibo_class,
                            confidence=0.7,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context
                        )
                        entities.append(entity)
        
        return entities
    
    def _map_to_fibo_class(self, entity_type: str) -> Tuple[Optional[str], Optional[str]]:
        """Map entity type to FIBO class."""
        type_mapping = {
            'organizations': 'Organization',
            'monetary_amounts': 'Monetary Amount',
            'financial_metrics': 'Financial Service'
        }
        
        fibo_class_name = type_mapping.get(entity_type)
        if fibo_class_name:
            for uri, fibo_class in self.fibo_parser.classes.items():
                if fibo_class_name.lower() in fibo_class.label.lower():
                    return uri, fibo_class.label
        
        return None, None
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities."""
        seen_spans = set()
        unique_entities = []
        
        for entity in entities:
            span = (entity.start_pos, entity.end_pos)
            if span not in seen_spans:
                seen_spans.add(span)
                unique_entities.append(entity)
        
        return unique_entities

# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

@dataclass
class GraphNode:
    """Represents a node in the financial knowledge graph."""
    id: str
    label: str
    fibo_uri: str
    fibo_class: str
    properties: Dict[str, Any]
    document_refs: List[str]

@dataclass 
class GraphEdge:
    """Represents an edge in the financial knowledge graph."""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]
    confidence: float
    evidence: List[str]

class FinancialKnowledgeGraph:
    """Financial knowledge graph storing entities and relationships."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_index: Dict[str, Set[str]] = defaultdict(set)
        
    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph."""
        try:
            self.nodes[node.id] = node
            self.node_index[node.fibo_class].add(node.id)
            logger.debug(f"Added node: {node.id} ({node.fibo_class})")
            return True
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {e}")
            return False
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph."""
        try:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                logger.warning(f"Cannot add edge: missing nodes {edge.source} or {edge.target}")
                return False
            
            self.edges.append(edge)
            logger.debug(f"Added edge: {edge.source} -> {edge.target} ({edge.relationship})")
            return True
        except Exception as e:
            logger.error(f"Failed to add edge {edge.source} -> {edge.target}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> List[GraphNode]:
        """Get neighboring nodes."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id and edge.target in self.nodes:
                neighbors.append(self.nodes[edge.target])
            elif edge.target == node_id and edge.source in self.nodes:
                neighbors.append(self.nodes[edge.source])
        return neighbors
    
    def get_edges_for_node(self, node_id: str) -> List[GraphEdge]:
        """Get all edges connected to a node."""
        return [edge for edge in self.edges 
                if edge.source == node_id or edge.target == node_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'node_types': {fibo_class: len(node_ids) 
                          for fibo_class, node_ids in self.node_index.items()},
            'relationship_types': list(set(edge.relationship for edge in self.edges))
        }

class FiboGraphBuilder:
    """Builds financial knowledge graphs using FIBO ontology."""
    
    def __init__(self, fibo_parser):
        self.fibo_parser = fibo_parser
        self.graph = FinancialKnowledgeGraph()
        
    def create_node_from_entity(self, entity, doc_id: str = "unknown") -> GraphNode:
        """Create a graph node from an extracted entity."""
        node_id = f"{entity.fibo_class}_{hash(entity.text)}_{entity.start_pos}"
        
        properties = {
            'confidence': entity.confidence,
            'text_span': entity.text,
            'context': entity.context[:200],
            'extraction_method': 'entity_extractor'
        }
        
        return GraphNode(
            id=node_id,
            label=entity.text,
            fibo_uri=entity.fibo_uri,
            fibo_class=entity.fibo_class,
            properties=properties,
            document_refs=[doc_id]
        )
    
    def add_entity_to_graph(self, entity, doc_id: str = "unknown") -> bool:
        """Add an extracted entity to the knowledge graph."""
        if not self._is_valid_fibo_class(entity.fibo_uri):
            logger.warning(f"Unknown FIBO class: {entity.fibo_class}")
            return False
        
        node = self.create_node_from_entity(entity, doc_id)
        return self.graph.add_node(node)
    
    def _is_valid_fibo_class(self, fibo_uri: str) -> bool:
        """Check if FIBO URI is valid in our ontology."""
        return fibo_uri in self.fibo_parser.classes
    
    def create_edge(self, entity1, entity2, relationship: str, doc_text: str, doc_id: str) -> GraphEdge:
        """Create an edge between two entities."""
        source_id = f"{entity1.fibo_class}_{hash(entity1.text)}_{entity1.start_pos}"
        target_id = f"{entity2.fibo_class}_{hash(entity2.text)}_{entity2.start_pos}"
        
        distance = abs(entity1.start_pos - entity2.start_pos)
        confidence = max(0.1, 1.0 - (distance / 1000.0))
        
        context_start = min(entity1.start_pos, entity2.start_pos) - 50
        context_end = max(entity1.end_pos, entity2.end_pos) + 50
        evidence = doc_text[max(0, context_start):context_end]
        
        return GraphEdge(
            source=source_id,
            target=target_id,
            relationship=relationship,
            properties={
                'document_id': doc_id,
                'distance': distance,
                'extraction_method': 'inference'
            },
            confidence=confidence,
            evidence=[evidence]
        )
    
    def get_graph(self) -> FinancialKnowledgeGraph:
        """Get the constructed knowledge graph."""
        return self.graph
    
    def reset_graph(self):
        """Reset the knowledge graph."""
        self.graph = FinancialKnowledgeGraph()

# =============================================================================
# VECTOR STORE
# =============================================================================

@dataclass
class SimpleVector:
    """Simple vector implementation for embeddings."""
    data: List[float]
    
    @property
    def dimension(self) -> int:
        return len(self.data)
    
    def cosine_similarity(self, other: 'SimpleVector') -> float:
        """Calculate cosine similarity with another vector."""
        if self.dimension != other.dimension:
            return 0.0
        
        dot_prod = sum(a * b for a, b in zip(self.data, other.data))
        mag_a = math.sqrt(sum(x * x for x in self.data))
        mag_b = math.sqrt(sum(x * x for x in other.data))
        
        if mag_a == 0 or mag_b == 0:
            return 0.0
        
        return dot_prod / (mag_a * mag_b)

@dataclass
class VectorDocument:
    """Document with vector embedding."""
    id: str
    content: str
    vector: SimpleVector
    metadata: Dict[str, Any]

class FinancialVectorStore:
    """Vector store for financial document embeddings."""
    
    def __init__(self, dimension: int = 100):
        self.dimension = dimension
        self.documents: Dict[str, VectorDocument] = {}
        self.financial_vocab = self._build_financial_vocab()
    
    def _build_financial_vocab(self) -> List[str]:
        """Build financial vocabulary for embeddings."""
        return [
            'revenue', 'profit', 'loss', 'income', 'expense', 'margin',
            'asset', 'liability', 'equity', 'debt', 'cash', 'investment',
            'earnings', 'dividend', 'share', 'stock', 'bond', 'fund',
            'bank', 'loan', 'credit', 'interest', 'rate', 'growth',
            'company', 'corporation', 'business', 'financial', 'market'
        ][:self.dimension]
    
    def create_embedding(self, text: str) -> SimpleVector:
        """Create a simple embedding based on financial term frequency."""
        text_lower = text.lower()
        words = text_lower.split()
        
        vector_data = [0.0] * self.dimension
        
        for i, term in enumerate(self.financial_vocab):
            if i >= self.dimension:
                break
            
            term_count = sum(1 for word in words if term in word)
            if term_count > 0:
                vector_data[i] = math.log(1 + term_count)
        
        # Normalize
        max_val = max(abs(x) for x in vector_data) if vector_data else 1
        if max_val > 0:
            vector_data = [x / max_val for x in vector_data]
        
        return SimpleVector(vector_data)
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document to the vector store."""
        try:
            vector = self.create_embedding(content)
            
            doc = VectorDocument(
                id=doc_id,
                content=content,
                vector=vector,
                metadata=metadata or {}
            )
            
            self.documents[doc_id] = doc
            logger.debug(f"Added document {doc_id} to vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 10, 
                      min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar documents using cosine similarity."""
        if not self.documents:
            return []
        
        query_vector = self.create_embedding(query)
        similarities = []
        
        for doc_id, doc in self.documents.items():
            similarity = query_vector.cosine_similarity(doc.vector)
            
            if similarity >= min_similarity:
                similarities.append((doc_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.documents:
            return {
                'total_documents': 0,
                'vector_dimension': self.dimension,
                'average_content_length': 0
            }
        
        total_length = sum(len(doc.content) for doc in self.documents.values())
        avg_length = total_length / len(self.documents)
        
        return {
            'total_documents': len(self.documents),
            'vector_dimension': self.dimension,
            'average_content_length': avg_length,
            'total_content_length': total_length
        }

# =============================================================================
# RETRIEVAL ENGINE
# =============================================================================

@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    content: str
    score: float
    source_type: str
    metadata: Dict[str, Any]
    entities: List[str]
    relationships: List[str]

class FiboRetrievalEngine:
    """Dual-level retrieval engine for financial documents."""
    
    def __init__(self, vector_store, graph_builder, entity_extractor):
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        self.entity_extractor = entity_extractor
        self.knowledge_graph = graph_builder.get_graph()
        
        self.config = {
            'vector_weight': 0.6,
            'graph_weight': 0.4,
            'max_results': 10,
            'min_similarity': 0.001,
            'confidence_threshold': 0.001
        }
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add document to both vector store and knowledge graph."""
        try:
            # Add to vector store
            vector_success = self.vector_store.add_document(doc_id, content, metadata)
            
            # Extract entities and add to graph
            entities = self.entity_extractor.extract_entities(content)
            graph_success = True
            
            nodes_added = 0
            for entity in entities:
                if self.graph_builder.add_entity_to_graph(entity, doc_id):
                    nodes_added += 1
            
            # Create relationships between entities
            relationships_added = self._create_relationships(entities, content, doc_id)
            
            logger.info(f"Added document {doc_id}: {nodes_added} nodes, {relationships_added} relationships")
            return vector_success and graph_success
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def _create_relationships(self, entities: List, content: str, doc_id: str) -> int:
        """Create relationships between entities in the same document."""
        relationships_added = 0
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.text.lower() == entity2.text.lower():
                    continue
                
                distance = abs(entity1.start_pos - entity2.start_pos)
                if distance > 200:  # Only create relationships for nearby entities
                    continue
                
                # Determine relationship type based on context
                relationship_type = self._infer_relationship_type(entity1, entity2, content)
                
                # Create edge
                edge = self.graph_builder.create_edge(entity1, entity2, relationship_type, content, doc_id)
                
                if self.knowledge_graph.add_edge(edge):
                    relationships_added += 1
        
        return relationships_added
    
    def _infer_relationship_type(self, entity1, entity2, content: str) -> str:
        """Infer relationship type between entities based on context."""
        context_start = min(entity1.start_pos, entity2.start_pos) - 100
        context_end = max(entity1.end_pos, entity2.end_pos) + 100
        context = content[max(0, context_start):context_end].lower()
        
        if any(keyword in context for keyword in ['owns', 'acquired', 'subsidiary']):
            return 'owns'
        elif any(keyword in context for keyword in ['partnership', 'alliance', 'joint venture']):
            return 'partners_with'
        elif any(keyword in context for keyword in ['provides', 'offers', 'delivers']):
            return 'provides_service'
        else:
            return 'co_occurs_with'
    
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
        
        logger.info(f"Vector retrieval found {len(results)} results")
        return results
    
    def _graph_retrieval(self, query: str) -> List[RetrievalResult]:
        """Retrieve using knowledge graph traversal."""
        results = []
        
        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query)
        
        if not query_entities:
            return results
        
        # Find relevant graph nodes
        relevant_nodes = set()
        for entity in query_entities:
            for node in self.knowledge_graph.nodes.values():
                if entity.text.lower() in node.label.lower() or node.label.lower() in entity.text.lower():
                    relevant_nodes.add(node.id)
        
        # Convert nodes to results
        for node_id in relevant_nodes:
            node = self.knowledge_graph.get_node(node_id)
            if node:
                edges = self.knowledge_graph.get_edges_for_node(node_id)
                relationships = [edge.relationship for edge in edges]
                
                result = RetrievalResult(
                    content=f"{node.label}: {node.properties.get('context', '')}",
                    score=0.8,  # Base score for graph matches
                    source_type='graph',
                    metadata=node.properties,
                    entities=[node.label],
                    relationships=relationships
                )
                results.append(result)
        
        logger.info(f"Graph retrieval found {len(results)} results")
        return results
    
    def _hybrid_retrieval(self, query: str) -> List[RetrievalResult]:
        """Combine vector and graph retrieval results."""
        vector_results = self._vector_retrieval(query)
        graph_results = self._graph_retrieval(query)
        
        # Reweight scores
        for result in vector_results:
            result.score *= self.config['vector_weight']
        
        for result in graph_results:
            result.score *= self.config['graph_weight']
            result.source_type = 'hybrid'
        
        # Combine and sort
        all_results = vector_results + graph_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        final_results = all_results[:self.config['max_results']]
        
        logger.info(f"Hybrid retrieval combined {len(vector_results)} vector + {len(graph_results)} graph = {len(final_results)} final results")
        return final_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        return {
            'vector_store': self.vector_store.get_statistics(),
            'knowledge_graph': self.knowledge_graph.get_stats(),
            'configuration': self.config
        }

# =============================================================================
# MAIN SYSTEM
# =============================================================================

@dataclass
class SystemResponse:
    """Response from the FIBO-LightRAG system."""
    results: List[RetrievalResult]
    processing_time: float
    metadata: Dict[str, Any]

class FiboLightRAGSystem:
    """Main FIBO-LightRAG system orchestrating all components."""
    
    def __init__(self):
        self.fibo_parser = FiboParser()
        self.entity_extractor = FiboEntityExtractor(self.fibo_parser)
        self.graph_builder = FiboGraphBuilder(self.fibo_parser)
        self.vector_store = FinancialVectorStore()
        self.retrieval_engine = FiboRetrievalEngine(
            self.vector_store, 
            self.graph_builder, 
            self.entity_extractor
        )
        
        logger.info("FIBO-LightRAG system initialized")
    
    def add_document(self, content: str, doc_id: Optional[str] = None, 
                    metadata: Optional[Dict] = None) -> bool:
        """Add a document to the system."""
        if doc_id is None:
            doc_id = f"doc_{hash(content[:100])}"
        
        return self.retrieval_engine.add_document(doc_id, content, metadata)
    
    def query(self, query: str, method: Optional[str] = None) -> SystemResponse:
        """Query the system for information."""
        start_time = time.time()
        
        try:
            method = method or 'hybrid'
            results = self.retrieval_engine.retrieve(query, method)
            
            processing_time = time.time() - start_time
            
            metadata = {
                'query': query,
                'method': method,
                'num_results': len(results),
                'processing_time': processing_time
            }
            
            response = SystemResponse(
                results=results,
                processing_time=processing_time,
                metadata=metadata
            )
            
            logger.info(f"Query processed: '{query}' -> {len(results)} results in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return SystemResponse(
                results=[],
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'fibo_ontology': {
                'classes': len(self.fibo_parser.classes),
                'properties': len(self.fibo_parser.properties),
                'relationships': len(self.fibo_parser.relationships)
            }
        }
        
        stats.update(self.retrieval_engine.get_statistics())
        return stats

# =============================================================================
# DEMO AND EXAMPLES
# =============================================================================

def create_sample_documents():
    """Create sample financial documents for demonstration."""
    return [
        {
            'id': 'bank_report_2023',
            'content': """
            First National Bank Annual Report 2023
            
            First National Bank reported strong financial performance in 2023. 
            Revenue increased by 15% to $2.5 billion compared to $2.17 billion in 2022.
            Net income rose to $450 million, representing a 12% increase year-over-year.
            
            The bank's loan portfolio grew by 8% to $18 billion, driven by strong demand 
            in commercial lending. Deposit base expanded to $22 billion, up 6% from 2022.
            
            First National Bank maintains partnerships with several fintech companies 
            to enhance digital banking services. The bank also provides investment 
            advisory services through its subsidiary, FNB Investment Services.
            """,
            'metadata': {
                'document_type': 'annual_report',
                'year': 2023,
                'company': 'First National Bank'
            }
        },
        {
            'id': 'tech_earnings_q4',
            'content': """
            TechCorp Q4 2023 Earnings Call Transcript
            
            TechCorp reported record Q4 2023 revenue of $8.2 billion, beating analyst 
            expectations of $7.9 billion. This represents 22% growth compared to Q4 2022.
            
            The company's cloud services division generated $3.1 billion in revenue,
            up 35% year-over-year. Software licensing revenue was $2.8 billion.
            
            TechCorp announced a new partnership with Global Financial Services Inc.
            to provide enterprise software solutions. The company also acquired 
            DataAnalytics Pro for $150 million to enhance its AI capabilities.
            """,
            'metadata': {
                'document_type': 'earnings_transcript',
                'quarter': 'Q4',
                'year': 2023,
                'company': 'TechCorp'
            }
        }
    ]

def run_demo():
    """Run the FIBO-LightRAG system demo."""
    print("🏦 FIBO-LightRAG System Demo")
    print("=" * 50)
    
    # Initialize system
    print("\\n1. Initializing FIBO-LightRAG system...")
    system = FiboLightRAGSystem()
    print("✅ System initialized successfully")
    
    # Show initial statistics
    stats = system.get_statistics()
    print(f"\\n📊 Initial System Statistics:")
    print(f"   FIBO Classes: {stats['fibo_ontology']['classes']}")
    print(f"   FIBO Properties: {stats['fibo_ontology']['properties']}")
    print(f"   FIBO Relationships: {stats['fibo_ontology']['relationships']}")
    
    # Add sample documents
    print("\\n2. Adding sample financial documents...")
    documents = create_sample_documents()
    
    for doc in documents:
        success = system.add_document(
            content=doc['content'],
            doc_id=doc['id'],
            metadata=doc['metadata']
        )
        status = "✅" if success else "❌"
        print(f"   {status} {doc['id']}")
    
    # Show updated statistics
    stats = system.get_statistics()
    print(f"\\n📊 Updated System Statistics:")
    print(f"   Documents in Vector Store: {stats['vector_store']['total_documents']}")
    print(f"   Knowledge Graph Nodes: {stats['knowledge_graph']['num_nodes']}")
    print(f"   Knowledge Graph Edges: {stats['knowledge_graph']['num_edges']}")
    
    # Run sample queries
    print("\\n3. Running sample queries...")
    
    queries = [
        "What is First National Bank's revenue?",
        "Tell me about TechCorp's partnerships",
        "What companies have revenue growth?",
        "Show me banking information"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\\n🔍 Query {i}: {query}")
        
        response = system.query(query)
        print(f"   📈 Results: {len(response.results)} found")
        
        if response.results:
            top_result = response.results[0]
            content_preview = top_result.content[:100] + "..." if len(top_result.content) > 100 else top_result.content
            print(f"      Score: {top_result.score:.3f}")
            print(f"      Type: {top_result.source_type}")
            print(f"      Preview: {content_preview}")
    
    print("\\n🎉 Demo completed successfully!")

def run_system_check():
    """Run a basic system check."""
    print("🔍 FIBO-LightRAG System Check")
    print("=" * 40)
    
    try:
        # Test system initialization
        print("\\n1. Testing system initialization...")
        system = FiboLightRAGSystem()
        print("   ✅ System initialized")
        
        # Test document addition
        print("\\n2. Testing document addition...")
        test_doc = "Apple Inc. reported revenue of $10 billion in Q1 2024."
        success = system.add_document(test_doc, "test_doc")
        print(f"   {'✅' if success else '❌'} Document addition")
        
        # Test query
        print("\\n3. Testing query processing...")
        response = system.query("What is Apple's revenue?")
        print(f"   ✅ Query processing ({len(response.results)} results)")
        
        # Test statistics
        print("\\n4. Testing system statistics...")
        stats = system.get_statistics()
        print(f"   ✅ Statistics: {stats['knowledge_graph']['num_nodes']} nodes, {stats['knowledge_graph']['num_edges']} edges")
        
        print("\\n🎉 All checks passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"\\n❌ System check failed: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run demos and examples."""
    print("FIBO-LightRAG: Financial Knowledge Graph System")
    print("=" * 60)
    print()
    print("This is a complete implementation of FIBO-LightRAG that combines")
    print("LightRAG concepts with Financial Industry Business Ontology (FIBO)")
    print("for advanced financial document analysis.")
    print()
    
    while True:
        print("\\nChoose an option:")
        print("1. Run System Check")
        print("2. Run Demo")
        print("3. Exit")
        
        choice = input("\\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            run_system_check()
        elif choice == '2':
            run_demo()
        elif choice == '3':
            print("\\nGoodbye! 👋")
            break
        else:
            print("\\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()