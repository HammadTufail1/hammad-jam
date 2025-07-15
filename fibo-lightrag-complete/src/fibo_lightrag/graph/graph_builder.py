"""
Knowledge graph construction using FIBO ontology for financial entities.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

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
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # fibo_class -> node_ids
        
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
            # Verify nodes exist
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
    
    def get_nodes_by_class(self, fibo_class: str) -> List[GraphNode]:
        """Get all nodes of a specific FIBO class."""
        node_ids = self.node_index.get(fibo_class, set())
        return [self.nodes[node_id] for node_id in node_ids]
    
    def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[GraphNode]:
        """Get neighboring nodes."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                if relationship_type is None or edge.relationship == relationship_type:
                    if edge.target in self.nodes:
                        neighbors.append(self.nodes[edge.target])
            elif edge.target == node_id:
                if relationship_type is None or edge.relationship == relationship_type:
                    if edge.source in self.nodes:
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
        self.inference_rules = self._build_inference_rules()
        
    def _build_inference_rules(self) -> Dict[str, List[Dict]]:
        """Build inference rules based on FIBO ontology."""
        rules = {
            'organizational_relationships': [
                {
                    'condition': ['Organization', 'Organization'],
                    'context_keywords': ['subsidiary', 'owns', 'acquired', 'merger'],
                    'relationship': 'owns'
                },
                {
                    'condition': ['Organization', 'Organization'],
                    'context_keywords': ['partnership', 'alliance', 'joint venture'],
                    'relationship': 'partners_with'
                }
            ],
            'service_relationships': [
                {
                    'condition': ['Organization', 'FinancialService'],
                    'context_keywords': ['provides', 'offers', 'delivers'],
                    'relationship': 'provides_service'
                }
            ],
            'product_relationships': [
                {
                    'condition': ['Organization', 'FinancialProduct'],
                    'context_keywords': ['issues', 'creates', 'underwrites'],
                    'relationship': 'issues_product'
                }
            ],
            'regulatory_relationships': [
                {
                    'condition': ['Organization', 'RegulatoryAgency'],
                    'context_keywords': ['regulated by', 'supervised by', 'reports to'],
                    'relationship': 'regulated_by'
                }
            ]
        }
        return rules
    
    def create_node_from_entity(self, entity, doc_id: str = "unknown") -> GraphNode:
        """Create a graph node from an extracted entity."""
        node_id = f"{entity.fibo_class}_{hash(entity.text)}_{entity.start_pos}"
        
        properties = {
            'confidence': entity.confidence,
            'text_span': entity.text,
            'context': entity.context[:200],  # Truncate context
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
        # Check if FIBO class is valid
        if not self._is_valid_fibo_class(entity.fibo_uri):
            logger.warning(f"Unknown FIBO class: {entity.fibo_class}")
            return False
        
        node = self.create_node_from_entity(entity, doc_id)
        return self.graph.add_node(node)
    
    def _is_valid_fibo_class(self, fibo_uri: str) -> bool:
        """Check if FIBO URI is valid in our ontology."""
        return fibo_uri in self.fibo_parser.classes
    
    def infer_relationships(self, entities: List, doc_text: str, doc_id: str = "unknown") -> int:
        """Infer relationships between entities using FIBO rules."""
        relationships_added = 0
        doc_text_lower = doc_text.lower()
        
        # Create entity pairs for relationship inference
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Skip if same entity
                if entity1.text == entity2.text:
                    continue
                
                # Check if entities are close in text (within 200 characters)
                distance = abs(entity1.start_pos - entity2.start_pos)
                if distance > 200:
                    continue
                
                # Try to infer relationship
                relationship = self._infer_relationship_type(
                    entity1, entity2, doc_text_lower
                )
                
                if relationship:
                    edge = self._create_edge(entity1, entity2, relationship, doc_text, doc_id)
                    if self.graph.add_edge(edge):
                        relationships_added += 1
        
        logger.info(f"Inferred {relationships_added} relationships from {len(entities)} entities")
        return relationships_added
    
    def _infer_relationship_type(self, entity1, entity2, doc_text: str) -> Optional[str]:
        """Infer relationship type between two entities."""
        # Get entity classes
        class1 = entity1.fibo_class
        class2 = entity2.fibo_class
        
        # Check inference rules
        for rule_type, rules in self.inference_rules.items():
            for rule in rules:
                condition = rule['condition']
                
                # Check if entity classes match rule condition
                if ((class1 in condition[0] and class2 in condition[1]) or
                    (class1 in condition[1] and class2 in condition[0])):
                    
                    # Check if context keywords are present
                    context_start = min(entity1.start_pos, entity2.start_pos) - 100
                    context_end = max(entity1.end_pos, entity2.end_pos) + 100
                    context = doc_text[max(0, context_start):context_end]
                    
                    for keyword in rule['context_keywords']:
                        if keyword in context:
                            return rule['relationship']
        
        # Default relationship for co-occurring entities
        return 'co_occurs_with'
    
    def _create_edge(self, entity1, entity2, relationship: str, doc_text: str, doc_id: str) -> GraphEdge:
        """Create an edge between two entities."""
        # Generate node IDs (same logic as in create_node_from_entity)
        source_id = f"{entity1.fibo_class}_{hash(entity1.text)}_{entity1.start_pos}"
        target_id = f"{entity2.fibo_class}_{hash(entity2.text)}_{entity2.start_pos}"
        
        # Calculate confidence based on entity proximity and context
        distance = abs(entity1.start_pos - entity2.start_pos)
        confidence = max(0.1, 1.0 - (distance / 1000.0))  # Closer entities = higher confidence
        
        # Extract evidence (context around entities)
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
    
    def perform_inference(self) -> int:
        """Apply inference rules to derive new relationships."""
        # Alias for backward compatibility
        return self.apply_inference_rules()
    
    def apply_inference_rules(self) -> int:
        """Apply FIBO-based inference rules to derive new relationships."""
        new_relationships = 0
        
        # Transitivity rules
        new_relationships += self._apply_transitivity_rules()
        
        # Hierarchy rules
        new_relationships += self._apply_hierarchy_rules()
        
        # Domain-specific rules
        new_relationships += self._apply_domain_rules()
        
        logger.info(f"Applied inference rules, added {new_relationships} new relationships")
        return new_relationships
    
    def _apply_transitivity_rules(self) -> int:
        """Apply transitivity rules (e.g., if A owns B and B owns C, then A owns C)."""
        new_relationships = 0
        ownership_edges = [edge for edge in self.graph.edges if edge.relationship == 'owns']
        
        for edge1 in ownership_edges:
            for edge2 in ownership_edges:
                if edge1.target == edge2.source and edge1.source != edge2.target:
                    # Check if transitive relationship already exists
                    exists = any(edge.source == edge1.source and edge.target == edge2.target 
                               and edge.relationship == 'owns' for edge in self.graph.edges)
                    
                    if not exists:
                        transitive_edge = GraphEdge(
                            source=edge1.source,
                            target=edge2.target,
                            relationship='owns',
                            properties={'inference_type': 'transitivity'},
                            confidence=min(edge1.confidence, edge2.confidence) * 0.8,
                            evidence=[f"Inferred from {edge1.source} -> {edge1.target} -> {edge2.target}"]
                        )
                        
                        if self.graph.add_edge(transitive_edge):
                            new_relationships += 1
        
        return new_relationships
    
    def _apply_hierarchy_rules(self) -> int:
        """Apply hierarchy rules based on FIBO class hierarchies."""
        new_relationships = 0
        
        # Use FIBO parent-child relationships to infer entity relationships
        for uri, fibo_class in self.fibo_parser.classes.items():
            for parent_uri in fibo_class.parent_classes:
                if parent_uri in self.fibo_parser.classes:
                    parent_class = self.fibo_parser.classes[parent_uri]
                    
                    # Find nodes of these classes
                    child_nodes = self.graph.get_nodes_by_class(fibo_class.label)
                    parent_nodes = self.graph.get_nodes_by_class(parent_class.label)
                    
                    # Create "is_a" relationships
                    for child_node in child_nodes:
                        for parent_node in parent_nodes:
                            if child_node.id != parent_node.id:
                                exists = any(edge.source == child_node.id and edge.target == parent_node.id 
                                           and edge.relationship == 'is_a' for edge in self.graph.edges)
                                
                                if not exists:
                                    is_a_edge = GraphEdge(
                                        source=child_node.id,
                                        target=parent_node.id,
                                        relationship='is_a',
                                        properties={'inference_type': 'hierarchy'},
                                        confidence=0.9,
                                        evidence=[f"FIBO hierarchy: {fibo_class.label} is_a {parent_class.label}"]
                                    )
                                    
                                    if self.graph.add_edge(is_a_edge):
                                        new_relationships += 1
        
        return new_relationships
    
    def _apply_domain_rules(self) -> int:
        """Apply domain-specific financial rules."""
        new_relationships = 0
        
        # Rule: Banks provide banking services
        banks = [node for node in self.graph.nodes.values() 
                if 'bank' in node.label.lower() or 'Bank' in node.fibo_class]
        
        services = [node for node in self.graph.nodes.values()
                   if 'Service' in node.fibo_class or 'service' in node.label.lower()]
        
        for bank in banks:
            for service in services:
                if 'banking' in service.label.lower() or 'financial' in service.label.lower():
                    exists = any(edge.source == bank.id and edge.target == service.id 
                               and edge.relationship == 'provides_service' for edge in self.graph.edges)
                    
                    if not exists:
                        service_edge = GraphEdge(
                            source=bank.id,
                            target=service.id,
                            relationship='provides_service',
                            properties={'inference_type': 'domain_rule'},
                            confidence=0.7,
                            evidence=["Domain rule: Banks typically provide banking services"]
                        )
                        
                        if self.graph.add_edge(service_edge):
                            new_relationships += 1
        
        return new_relationships
    
    def get_graph(self) -> FinancialKnowledgeGraph:
        """Get the constructed knowledge graph."""
        return self.graph
    
    def reset_graph(self):
        """Reset the knowledge graph."""
        self.graph = FinancialKnowledgeGraph()
        logger.info("Knowledge graph reset")