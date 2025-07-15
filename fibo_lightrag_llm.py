#!/usr/bin/env python3
"""
FIBO-LightRAG: LLM-Powered Financial Knowledge Graph System

A comprehensive implementation using real LLMs and embeddings for:
- Entity extraction with FIBO ontology mapping
- Relationship inference and knowledge graph construction  
- Intelligent query processing and understanding
- Dual-level retrieval with semantic embeddings

Supports multiple LLM providers: OpenAI, Anthropic, Google, Local models

Dependencies:
    pip install openai anthropic google-generativeai sentence-transformers numpy

Usage:
    python fibo_lightrag_llm.py

Author: FIBO-LightRAG Development Team
Version: 2.0.0 (LLM-Powered)
"""

import json
import logging
import os
import time
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from abc import ABC, abstractmethod
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# LLM PROVIDERS ABSTRACTION
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Generate text completion."""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT and embeddings."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.embedding_model = embedding_model
            logger.info(f"Initialized OpenAI provider with {model}")
        except ImportError:
            raise ImportError("Please install: pip install openai")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI text generation failed: {e}")
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        try:
            # Truncate text to avoid token limits
            text = text[:8000]  # Safe limit for most embedding models
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            return [0.0] * 1536  # Default embedding size

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            logger.info(f"Initialized Anthropic provider with {model}")
        except ImportError:
            raise ImportError("Please install: pip install anthropic")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic text generation failed: {e}")
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        # Anthropic doesn't provide embeddings, fall back to sentence transformers
        return self._get_sentence_transformer_embedding(text)
    
    def _get_sentence_transformer_embedding(self, text: str) -> List[float]:
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embedding = self._embedding_model.encode(text)
            return embedding.tolist()
        except ImportError:
            logger.warning("Sentence transformers not available, using dummy embedding")
            return [0.1] * 384

class LocalLLMProvider(LLMProvider):
    """Local LLM provider using Ollama or similar."""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized Local LLM provider with {model}")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        try:
            import requests
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Local LLM request failed: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        # Use sentence transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embedding = self._embedding_model.encode(text)
            return embedding.tolist()
        except ImportError:
            logger.warning("Sentence transformers not available, using dummy embedding")
            return [0.1] * 384

# =============================================================================
# FIBO ONTOLOGY WITH LLM ENHANCEMENT
# =============================================================================

@dataclass
class FiboClass:
    """Enhanced FIBO class with LLM-generated descriptions."""
    uri: str
    label: str
    definition: str
    parent_classes: List[str]
    properties: List[str]
    llm_description: str = ""
    example_entities: List[str] = None

@dataclass
class FiboProperty:
    """Enhanced FIBO property."""
    uri: str
    label: str
    definition: str
    domain: List[str]
    range: List[str]

@dataclass
class FiboRelationship:
    """FIBO relationship with confidence scoring."""
    subject: str
    predicate: str
    object: str
    relationship_type: str
    confidence: float = 1.0

class LLMEnhancedFiboParser:
    """FIBO parser enhanced with LLM understanding."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.classes: Dict[str, FiboClass] = {}
        self.properties: Dict[str, FiboProperty] = {}
        self.relationships: List[FiboRelationship] = []
        self._initialize_core_fibo_ontology()
    
    def _initialize_core_fibo_ontology(self):
        """Initialize core FIBO ontology with LLM enhancements."""
        logger.info("Initializing FIBO ontology with LLM enhancements...")
        
        # Core FIBO classes
        core_classes = {
            "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/Organization": {
                "label": "Organization",
                "definition": "A formal or informal organization of people or other legal entities",
                "parent_classes": [],
                "properties": []
            },
            "https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/FinancialInstitution": {
                "label": "Financial Institution", 
                "definition": "An organization that provides financial services to customers",
                "parent_classes": ["https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/Organization"],
                "properties": []
            },
            "https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/Bank": {
                "label": "Bank",
                "definition": "A financial institution that accepts deposits and creates credit",
                "parent_classes": ["https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/FinancialInstitution"],
                "properties": []
            },
            "https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/FinancialInstrument": {
                "label": "Financial Instrument",
                "definition": "A tradable asset, security, or contract",
                "parent_classes": [],
                "properties": []
            },
            "https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/MonetaryAmount": {
                "label": "Monetary Amount",
                "definition": "A number of monetary units specified in a currency",
                "parent_classes": [],
                "properties": []
            },
            "https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/FinancialService": {
                "label": "Financial Service",
                "definition": "A service provided by the finance industry",
                "parent_classes": [],
                "properties": []
            }
        }
        
        # Enhance each class with LLM
        for uri, class_data in core_classes.items():
            enhanced_description = self._generate_class_description(class_data["label"], class_data["definition"])
            example_entities = self._generate_example_entities(class_data["label"])
            
            self.classes[uri] = FiboClass(
                uri=uri,
                label=class_data["label"],
                definition=class_data["definition"],
                parent_classes=class_data["parent_classes"],
                properties=class_data["properties"],
                llm_description=enhanced_description,
                example_entities=example_entities
            )
        
        logger.info(f"Initialized {len(self.classes)} FIBO classes with LLM enhancements")
    
    def _generate_class_description(self, label: str, definition: str) -> str:
        """Generate enhanced description using LLM."""
        prompt = f"""
        For the FIBO (Financial Industry Business Ontology) class "{label}" with definition:
        "{definition}"
        
        Provide a comprehensive description that includes:
        1. What this class represents in financial contexts
        2. Key characteristics and attributes
        3. How it relates to other financial entities
        4. Common use cases in financial analysis
        
        Keep it concise but informative (2-3 sentences).
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=200, temperature=0.1)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate class description for {label}: {e}")
            return definition
    
    def _generate_example_entities(self, label: str) -> List[str]:
        """Generate example entities for a FIBO class."""
        prompt = f"""
        For the FIBO class "{label}", provide 5-7 realistic examples of entities that would belong to this class.
        
        Examples should be:
        - Real or realistic names/entities
        - Relevant to financial industry
        - Diverse in scope
        
        Return only the entity names, one per line, no explanations.
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=150, temperature=0.3)
            examples = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return examples[:7]  # Limit to 7 examples
        except Exception as e:
            logger.error(f"Failed to generate examples for {label}: {e}")
            return []

# =============================================================================
# LLM-POWERED ENTITY EXTRACTION
# =============================================================================

@dataclass
class ExtractedEntity:
    """Entity extracted using LLM with FIBO mapping."""
    text: str
    fibo_uri: str
    fibo_class: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    llm_reasoning: str = ""

class LLMEntityExtractor:
    """LLM-powered entity extraction with FIBO ontology mapping."""
    
    def __init__(self, llm_provider: LLMProvider, fibo_parser: LLMEnhancedFiboParser):
        self.llm = llm_provider
        self.fibo_parser = fibo_parser
        
    def extract_entities(self, text: str, max_entities: int = 20) -> List[ExtractedEntity]:
        """Extract entities using LLM with FIBO classification."""
        logger.info("Extracting entities using LLM...")
        
        # Create FIBO classes context for the LLM
        fibo_context = self._create_fibo_context()
        
        prompt = f"""
        You are a financial document analysis expert. Extract financial entities from the following text and classify them according to FIBO (Financial Industry Business Ontology) classes.

        FIBO Classes Available:
        {fibo_context}

        Document Text:
        {text}

        Instructions:
        1. Identify financial entities (organizations, amounts, instruments, services, etc.)
        2. For each entity, determine the most appropriate FIBO class
        3. Provide confidence score (0.0-1.0) for each classification
        4. Include reasoning for the classification

        Return ONLY a JSON array with this format:
        [
            {{
                "text": "entity text",
                "fibo_class": "exact FIBO class label",
                "confidence": 0.95,
                "reasoning": "why this classification"
            }}
        ]

        Limit to {max_entities} most important entities.
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=2000, temperature=0.1)
            entities = self._parse_llm_entity_response(response, text)
            
            logger.info(f"LLM extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return []
    
    def _create_fibo_context(self) -> str:
        """Create context string describing available FIBO classes."""
        context_parts = []
        
        for uri, fibo_class in self.fibo_parser.classes.items():
            class_info = f"- {fibo_class.label}: {fibo_class.definition}"
            if fibo_class.example_entities:
                examples = ", ".join(fibo_class.example_entities[:3])
                class_info += f" (Examples: {examples})"
            context_parts.append(class_info)
        
        return "\\n".join(context_parts)
    
    def _parse_llm_entity_response(self, response: str, original_text: str) -> List[ExtractedEntity]:
        """Parse LLM response into ExtractedEntity objects."""
        entities = []
        
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                entity_data = json.loads(json_str)
                
                for item in entity_data:
                    entity = self._create_entity_from_llm_data(item, original_text)
                    if entity:
                        entities.append(entity)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM entity response as JSON: {e}")
            # Fallback: try to extract entities using regex from the response
            entities = self._fallback_entity_extraction(response, original_text)
        
        return entities
    
    def _create_entity_from_llm_data(self, item: Dict, original_text: str) -> Optional[ExtractedEntity]:
        """Create ExtractedEntity from LLM-parsed data."""
        try:
            entity_text = item.get("text", "").strip()
            fibo_class_label = item.get("fibo_class", "").strip()
            confidence = float(item.get("confidence", 0.5))
            reasoning = item.get("reasoning", "")
            
            if not entity_text or not fibo_class_label:
                return None
            
            # Find entity position in original text
            start_pos = original_text.lower().find(entity_text.lower())
            if start_pos == -1:
                start_pos = 0
            
            end_pos = start_pos + len(entity_text)
            
            # Get context around entity
            context_start = max(0, start_pos - 100)
            context_end = min(len(original_text), end_pos + 100)
            context = original_text[context_start:context_end]
            
            # Map FIBO class label to URI
            fibo_uri = self._find_fibo_uri_by_label(fibo_class_label)
            if not fibo_uri:
                logger.warning(f"Could not find FIBO URI for class: {fibo_class_label}")
                return None
            
            return ExtractedEntity(
                text=entity_text,
                fibo_uri=fibo_uri,
                fibo_class=fibo_class_label,
                confidence=confidence,
                start_pos=start_pos,
                end_pos=end_pos,
                context=context,
                llm_reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Failed to create entity from LLM data: {e}")
            return None
    
    def _find_fibo_uri_by_label(self, label: str) -> Optional[str]:
        """Find FIBO URI by class label."""
        for uri, fibo_class in self.fibo_parser.classes.items():
            if fibo_class.label.lower() == label.lower():
                return uri
        return None
    
    def _fallback_entity_extraction(self, response: str, original_text: str) -> List[ExtractedEntity]:
        """Fallback entity extraction when JSON parsing fails."""
        entities = []
        
        # Simple fallback: look for entity mentions in the response
        lines = response.split('\\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['bank', 'company', 'corporation', '$', 'million', 'billion']):
                # Try to extract entity from line
                words = line.split()
                for word in words:
                    if len(word) > 3 and word[0].isupper():
                        # Simple heuristic for entity
                        start_pos = original_text.find(word)
                        if start_pos >= 0:
                            entity = ExtractedEntity(
                                text=word,
                                fibo_uri="https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/Organization",
                                fibo_class="Organization",
                                confidence=0.3,  # Low confidence for fallback
                                start_pos=start_pos,
                                end_pos=start_pos + len(word),
                                context=line,
                                llm_reasoning="Fallback extraction"
                            )
                            entities.append(entity)
                            break
        
        return entities[:5]  # Limit fallback entities

# =============================================================================
# LLM-POWERED KNOWLEDGE GRAPH
# =============================================================================

@dataclass
class GraphNode:
    """Knowledge graph node with LLM enhancements."""
    id: str
    label: str
    fibo_uri: str
    fibo_class: str
    properties: Dict[str, Any]
    document_refs: List[str]
    llm_summary: str = ""

@dataclass
class GraphEdge:
    """Knowledge graph edge with LLM-inferred relationships."""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]
    confidence: float
    evidence: List[str]
    llm_reasoning: str = ""

class LLMKnowledgeGraph:
    """LLM-enhanced knowledge graph for financial entities."""
    
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
        """Get node by ID."""
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
        """Get all edges for a node."""
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

class LLMGraphBuilder:
    """LLM-powered knowledge graph builder."""
    
    def __init__(self, llm_provider: LLMProvider, fibo_parser: LLMEnhancedFiboParser):
        self.llm = llm_provider
        self.fibo_parser = fibo_parser
        self.graph = LLMKnowledgeGraph()
    
    def create_node_from_entity(self, entity: ExtractedEntity, doc_id: str) -> GraphNode:
        """Create enhanced graph node from entity."""
        node_id = f"{entity.fibo_class}_{hashlib.md5(entity.text.encode()).hexdigest()[:8]}"
        
        # Generate LLM summary for the entity
        llm_summary = self._generate_entity_summary(entity)
        
        properties = {
            'confidence': entity.confidence,
            'text_span': entity.text,
            'context': entity.context[:300],
            'extraction_method': 'llm',
            'llm_reasoning': entity.llm_reasoning
        }
        
        return GraphNode(
            id=node_id,
            label=entity.text,
            fibo_uri=entity.fibo_uri,
            fibo_class=entity.fibo_class,
            properties=properties,
            document_refs=[doc_id],
            llm_summary=llm_summary
        )
    
    def _generate_entity_summary(self, entity: ExtractedEntity) -> str:
        """Generate LLM summary for an entity."""
        prompt = f"""
        For the financial entity "{entity.text}" classified as {entity.fibo_class}:
        
        Context: {entity.context}
        
        Provide a brief 1-2 sentence summary of what this entity represents and its significance in the financial context.
        Focus on factual information that can be inferred from the context.
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=100, temperature=0.1)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate entity summary: {e}")
            return f"{entity.fibo_class} entity: {entity.text}"
    
    def add_entity_to_graph(self, entity: ExtractedEntity, doc_id: str) -> bool:
        """Add entity to knowledge graph."""
        node = self.create_node_from_entity(entity, doc_id)
        return self.graph.add_node(node)
    
    def infer_relationships_llm(self, entities: List[ExtractedEntity], doc_text: str, doc_id: str) -> int:
        """Use LLM to infer relationships between entities."""
        if len(entities) < 2:
            return 0
        
        logger.info(f"Using LLM to infer relationships between {len(entities)} entities...")
        
        # Create entity context for LLM
        entity_context = self._create_entity_context(entities)
        
        prompt = f"""
        Analyze the following financial document text and identify relationships between the extracted entities.

        Entities found:
        {entity_context}

        Document context:
        {doc_text[:2000]}...

        For each pair of entities that have a meaningful relationship, identify:
        1. The type of relationship (owns, partners_with, provides_service, competes_with, subsidiary_of, regulates, etc.)
        2. Confidence level (0.0-1.0)
        3. Evidence from the text

        Return ONLY a JSON array:
        [
            {{
                "source_entity": "Entity 1 text",
                "target_entity": "Entity 2 text", 
                "relationship_type": "relationship name",
                "confidence": 0.85,
                "evidence": "text evidence for this relationship"
            }}
        ]

        Only include relationships with confidence > 0.5.
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=1500, temperature=0.1)
            relationships = self._parse_llm_relationships(response, entities, doc_text, doc_id)
            
            relationships_added = 0
            for rel_data in relationships:
                edge = self._create_edge_from_llm_data(rel_data, entities, doc_text, doc_id)
                if edge and self.graph.add_edge(edge):
                    relationships_added += 1
            
            logger.info(f"LLM inferred {relationships_added} relationships")
            return relationships_added
            
        except Exception as e:
            logger.error(f"LLM relationship inference failed: {e}")
            return 0
    
    def _create_entity_context(self, entities: List[ExtractedEntity]) -> str:
        """Create context string for entities."""
        context_parts = []
        for i, entity in enumerate(entities, 1):
            context_parts.append(f"{i}. {entity.text} ({entity.fibo_class})")
        return "\\n".join(context_parts)
    
    def _parse_llm_relationships(self, response: str, entities: List[ExtractedEntity], 
                                doc_text: str, doc_id: str) -> List[Dict]:
        """Parse LLM relationship response."""
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM relationship response: {e}")
        
        return []
    
    def _create_edge_from_llm_data(self, rel_data: Dict, entities: List[ExtractedEntity], 
                                  doc_text: str, doc_id: str) -> Optional[GraphEdge]:
        """Create graph edge from LLM relationship data."""
        try:
            source_text = rel_data.get("source_entity", "").strip()
            target_text = rel_data.get("target_entity", "").strip()
            relationship = rel_data.get("relationship_type", "").strip()
            confidence = float(rel_data.get("confidence", 0.5))
            evidence = rel_data.get("evidence", "")
            
            # Find source and target entities
            source_entity = None
            target_entity = None
            
            for entity in entities:
                if entity.text.lower() == source_text.lower():
                    source_entity = entity
                elif entity.text.lower() == target_text.lower():
                    target_entity = entity
            
            if not source_entity or not target_entity:
                logger.warning(f"Could not find entities for relationship: {source_text} -> {target_text}")
                return None
            
            # Create node IDs
            source_id = f"{source_entity.fibo_class}_{hashlib.md5(source_entity.text.encode()).hexdigest()[:8]}"
            target_id = f"{target_entity.fibo_class}_{hashlib.md5(target_entity.text.encode()).hexdigest()[:8]}"
            
            return GraphEdge(
                source=source_id,
                target=target_id,
                relationship=relationship,
                properties={
                    'document_id': doc_id,
                    'extraction_method': 'llm'
                },
                confidence=confidence,
                evidence=[evidence],
                llm_reasoning=f"LLM inferred {relationship} relationship with {confidence:.2f} confidence"
            )
            
        except Exception as e:
            logger.error(f"Failed to create edge from LLM data: {e}")
            return None
    
    def get_graph(self) -> LLMKnowledgeGraph:
        """Get the knowledge graph."""
        return self.graph
    
    def reset_graph(self):
        """Reset the knowledge graph."""
        self.graph = LLMKnowledgeGraph()

# =============================================================================
# LLM-POWERED VECTOR STORE  
# =============================================================================

@dataclass
class VectorDocument:
    """Document with real neural embeddings."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]

class LLMVectorStore:
    """Vector store using real neural embeddings."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.documents: Dict[str, VectorDocument] = {}
        
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add document with neural embedding."""
        try:
            # Generate real embedding using LLM provider
            embedding = self.llm.generate_embedding(content)
            
            doc = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            self.documents[doc_id] = doc
            logger.debug(f"Added document {doc_id} with {len(embedding)}-dim embedding")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 10, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """Search using cosine similarity of neural embeddings."""
        if not self.documents:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.llm.generate_embedding(query)
            
            similarities = []
            for doc_id, doc in self.documents.items():
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                
                if similarity >= min_similarity:
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import math
            
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.documents:
            return {
                'total_documents': 0,
                'embedding_dimension': 0,
                'average_content_length': 0
            }
        
        sample_doc = next(iter(self.documents.values()))
        total_length = sum(len(doc.content) for doc in self.documents.values())
        
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': len(sample_doc.embedding),
            'average_content_length': total_length / len(self.documents),
            'total_content_length': total_length
        }

# =============================================================================
# COMPLETE LLM SYSTEM INTEGRATION
# =============================================================================

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with LLM insights."""
    content: str
    score: float
    source_type: str
    metadata: Dict[str, Any]
    entities: List[str]
    relationships: List[str]
    llm_summary: str = ""

@dataclass
class SystemResponse:
    """Complete system response with LLM analysis."""
    results: List[RetrievalResult]
    processing_time: float
    metadata: Dict[str, Any]
    llm_analysis: str = ""

class FiboLightRAGLLMSystem:
    """Complete FIBO-LightRAG system powered by LLMs."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.fibo_parser = LLMEnhancedFiboParser(llm_provider)
        self.entity_extractor = LLMEntityExtractor(llm_provider, self.fibo_parser)
        self.graph_builder = LLMGraphBuilder(llm_provider, self.fibo_parser)
        self.vector_store = LLMVectorStore(llm_provider)
        
        # Configuration
        self.config = {
            'vector_weight': 0.6,
            'graph_weight': 0.4,
            'max_results': 10,
            'min_similarity': 0.1,
            'confidence_threshold': 0.3
        }
        
        logger.info("LLM-powered FIBO-LightRAG system initialized")
    
    def add_document(self, content: str, doc_id: Optional[str] = None, 
                    metadata: Optional[Dict] = None) -> bool:
        """Add document with full LLM processing."""
        if doc_id is None:
            doc_id = f"doc_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        logger.info(f"Processing document {doc_id} with LLM...")
        
        try:
            # Add to vector store with neural embeddings
            vector_success = self.vector_store.add_document(doc_id, content, metadata)
            
            # Extract entities using LLM
            entities = self.entity_extractor.extract_entities(content)
            
            # Add entities to knowledge graph
            nodes_added = 0
            for entity in entities:
                if self.graph_builder.add_entity_to_graph(entity, doc_id):
                    nodes_added += 1
            
            # Infer relationships using LLM
            relationships_added = self.graph_builder.infer_relationships_llm(entities, content, doc_id)
            
            logger.info(f"Document {doc_id}: {nodes_added} entities, {relationships_added} relationships")
            return vector_success and nodes_added > 0
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            return False
    
    def query(self, query: str, method: str = 'hybrid') -> SystemResponse:
        """Query with full LLM analysis."""
        start_time = time.time()
        
        logger.info(f"Processing query with LLM: {query}")
        
        try:
            # Get retrieval results
            if method == 'vector':
                results = self._vector_retrieval(query)
            elif method == 'graph':
                results = self._graph_retrieval(query)
            else:  # hybrid
                results = self._hybrid_retrieval(query)
            
            # Generate LLM analysis of results
            llm_analysis = self._generate_query_analysis(query, results)
            
            processing_time = time.time() - start_time
            
            response = SystemResponse(
                results=results,
                processing_time=processing_time,
                metadata={
                    'query': query,
                    'method': method,
                    'num_results': len(results),
                    'processing_time': processing_time
                },
                llm_analysis=llm_analysis
            )
            
            logger.info(f"Query completed: {len(results)} results in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return SystemResponse(
                results=[],
                processing_time=time.time() - start_time,
                metadata={'error': str(e)},
                llm_analysis="Query processing failed due to an error."
            )
    
    def _vector_retrieval(self, query: str) -> List[RetrievalResult]:
        """Vector retrieval with neural embeddings."""
        results = []
        
        similar_docs = self.vector_store.search_similar(
            query,
            top_k=self.config['max_results'],
            min_similarity=self.config['min_similarity']
        )
        
        for doc_id, similarity in similar_docs:
            doc = self.vector_store.get_document(doc_id)
            if doc and similarity >= self.config['confidence_threshold']:
                # Generate LLM summary for this result
                summary = self._generate_result_summary(query, doc.content)
                
                result = RetrievalResult(
                    content=doc.content,
                    score=similarity,
                    source_type='vector',
                    metadata=doc.metadata,
                    entities=[],
                    relationships=[],
                    llm_summary=summary
                )
                results.append(result)
        
        return results
    
    def _graph_retrieval(self, query: str) -> List[RetrievalResult]:
        """Graph retrieval with LLM entity matching."""
        results = []
        
        # Extract entities from query using LLM
        query_entities = self.entity_extractor.extract_entities(query)
        
        if not query_entities:
            return results
        
        # Find matching nodes in knowledge graph
        relevant_nodes = set()
        for query_entity in query_entities:
            for node in self.graph_builder.graph.nodes.values():
                if (query_entity.text.lower() in node.label.lower() or
                    node.label.lower() in query_entity.text.lower()):
                    relevant_nodes.add(node.id)
        
        # Convert nodes to results
        for node_id in relevant_nodes:
            node = self.graph_builder.graph.get_node(node_id)
            if node:
                edges = self.graph_builder.graph.get_edges_for_node(node_id)
                relationships = [edge.relationship for edge in edges]
                
                result = RetrievalResult(
                    content=f"{node.label}: {node.llm_summary}",
                    score=0.8,  # Base score for graph matches
                    source_type='graph',
                    metadata=node.properties,
                    entities=[node.label],
                    relationships=relationships,
                    llm_summary=node.llm_summary
                )
                results.append(result)
        
        return results
    
    def _hybrid_retrieval(self, query: str) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector and graph."""
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
        
        return all_results[:self.config['max_results']]
    
    def _generate_result_summary(self, query: str, content: str) -> str:
        """Generate LLM summary for a retrieval result."""
        prompt = f"""
        For the user query: "{query}"
        
        Summarize how the following content answers or relates to the query.
        Focus on the most relevant information.
        
        Content:
        {content[:1000]}...
        
        Provide a concise 2-3 sentence summary.
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=150, temperature=0.1)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate result summary: {e}")
            return "Content relevant to query."
    
    def _generate_query_analysis(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate overall analysis of query results."""
        if not results:
            return "No relevant results found for the query."
        
        # Create results summary for LLM
        results_summary = []
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            summary = f"{i}. {result.llm_summary or result.content[:100]}... (Score: {result.score:.2f})"
            results_summary.append(summary)
        
        prompt = f"""
        Analyze the search results for the financial query: "{query}"
        
        Results found:
        {chr(10).join(results_summary)}
        
        Provide a comprehensive analysis including:
        1. How well the results answer the query
        2. Key insights from the financial information
        3. Any patterns or trends identified
        4. Recommendations for further analysis
        
        Keep it concise but informative (3-4 sentences).
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=300, temperature=0.1)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate query analysis: {e}")
            return f"Found {len(results)} results relevant to the financial query."
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'fibo_ontology': {
                'classes': len(self.fibo_parser.classes),
                'properties': len(self.fibo_parser.properties),
                'relationships': len(self.fibo_parser.relationships)
            },
            'vector_store': self.vector_store.get_statistics(),
            'knowledge_graph': self.graph_builder.graph.get_stats(),
            'configuration': self.config
        }

# =============================================================================
# PROVIDER FACTORY AND DEMO
# =============================================================================

class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        """Create LLM provider by type."""
        if provider_type.lower() == 'openai':
            api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            return OpenAIProvider(api_key, kwargs.get('model', 'gpt-4o-mini'))
        
        elif provider_type.lower() == 'anthropic':
            api_key = kwargs.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
            return AnthropicProvider(api_key, kwargs.get('model', 'claude-3-sonnet-20240229'))
        
        elif provider_type.lower() == 'local':
            return LocalLLMProvider(kwargs.get('model', 'llama3'))
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

def create_sample_financial_documents():
    """Create sample financial documents for testing."""
    return [
        {
            'id': 'jpmorgan_2023_annual',
            'content': """
            JPMorgan Chase & Co. 2023 Annual Report - Executive Summary
            
            JPMorgan Chase reported record full-year 2023 net income of $49.6 billion, or $15.92 per share. 
            Net revenue was $162.4 billion for the year, up from $140.8 billion in 2022.
            
            The Consumer & Community Banking segment earned $5.3 billion in net income on revenue of $55.8 billion.
            Our Corporate & Investment Bank delivered net income of $15.2 billion on revenue of $45.1 billion.
            The Commercial Banking division generated $6.5 billion in net income.
            
            JPMorgan maintained its position as the largest bank in the United States by assets, with total assets 
            of $3.7 trillion at year-end 2023. The bank's Common Equity Tier 1 (CET1) ratio was 15.0%, 
            well above regulatory requirements.
            
            During 2023, JPMorgan completed the acquisition of First Republic Bank following its failure, 
            adding approximately $229 billion in assets. The bank also expanded its partnership with 
            Amazon Web Services to enhance digital banking capabilities.
            
            JPMorgan Chase returned $29.2 billion to shareholders through dividends and share buybacks in 2023.
            The bank serves over 66 million consumer households and 5 million small businesses across 
            4,800 branches in 48 states.
            """,
            'metadata': {
                'company': 'JPMorgan Chase',
                'document_type': 'annual_report',
                'year': 2023,
                'filing_type': '10-K'
            }
        },
        {
            'id': 'goldman_q4_2023_earnings',
            'content': """
            Goldman Sachs Group Inc. Fourth Quarter 2023 Earnings Report
            
            Goldman Sachs reported fourth quarter net revenues of $10.87 billion, compared to $10.59 billion 
            in the fourth quarter of 2022. Net earnings were $2.01 billion, or $5.48 per diluted share.
            
            Investment Banking revenues were $2.05 billion in the fourth quarter, up 35% from the prior year, 
            reflecting higher revenues across equity and debt underwriting. Global Markets generated $3.92 billion 
            in net revenues for the quarter.
            
            Asset & Wealth Management reported net revenues of $3.72 billion in the fourth quarter. 
            The division's assets under supervision totaled $2.93 trillion at year-end.
            
            Goldman Sachs maintains strategic partnerships with Apple for the Apple Card program and 
            continues to expand Marcus, its digital banking platform. The firm is also exploring 
            opportunities in cryptocurrency and blockchain technologies.
            
            The firm's return on equity (ROE) was 8.3% for the full year 2023. Book value per share 
            increased to $295.41 at the end of 2023. Goldman Sachs employs approximately 45,000 people 
            globally across 30 countries.
            
            Looking ahead, Goldman Sachs is focused on enhancing its technology infrastructure and 
            expanding its presence in Asia-Pacific markets, particularly in wealth management services.
            """,
            'metadata': {
                'company': 'Goldman Sachs',
                'document_type': 'earnings_report',
                'quarter': 'Q4',
                'year': 2023
            }
        },
        {
            'id': 'microsoft_cloud_expansion',
            'content': """
            Microsoft Corporation - Cloud Services Financial Update
            
            Microsoft's Intelligent Cloud segment delivered record quarterly revenue of $25.9 billion, 
            representing 20% growth year-over-year. Azure and other cloud services revenue increased 30%, 
            driven by strong demand for AI and machine learning capabilities.
            
            The company has invested heavily in artificial intelligence partnerships, most notably 
            with OpenAI, which has resulted in significant competitive advantages in the enterprise market. 
            Microsoft's AI services are now integrated across its entire product portfolio.
            
            Enterprise customers are increasingly adopting Microsoft's cloud solutions, with companies 
            like JPMorgan Chase and Goldman Sachs expanding their use of Azure for critical financial 
            applications. The partnership with financial institutions has been particularly strong 
            in the areas of data analytics and regulatory compliance.
            
            Microsoft Cloud revenue reached $33.7 billion for the quarter, accounting for over 50% 
            of total company revenue. The company's subscription-based model provides predictable 
            recurring revenue streams and high customer retention rates.
            
            Looking forward, Microsoft is investing $80 billion in AI infrastructure and has committed 
            to carbon neutrality by 2030. The company continues to expand its global data center 
            footprint to support growing demand for cloud services.
            """,
            'metadata': {
                'company': 'Microsoft',
                'document_type': 'financial_update',
                'focus': 'cloud_services',
                'year': 2023
            }
        }
    ]

def run_llm_demo():
    """Run demo of LLM-powered FIBO-LightRAG system."""
    print("🤖 LLM-Powered FIBO-LightRAG System Demo")
    print("=" * 60)
    
    # Provider selection
    print("\\nSelect LLM provider:")
    print("1. OpenAI (requires API key)")
    print("2. Anthropic (requires API key)")  
    print("3. Local LLM (requires Ollama)")
    print("4. Demo mode (limited functionality)")
    
    choice = input("\\nEnter choice (1-4): ").strip()
    
    try:
        if choice == '1':
            api_key = input("Enter OpenAI API key (or press Enter to use OPENAI_API_KEY env var): ").strip()
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("❌ OpenAI API key required")
                return
            
            provider = OpenAIProvider(api_key)
            print("✅ OpenAI provider initialized")
            
        elif choice == '2':
            api_key = input("Enter Anthropic API key (or press Enter to use ANTHROPIC_API_KEY env var): ").strip()
            if not api_key:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("❌ Anthropic API key required")
                return
            
            provider = AnthropicProvider(api_key)
            print("✅ Anthropic provider initialized")
            
        elif choice == '3':
            provider = LocalLLMProvider()
            print("✅ Local LLM provider initialized")
            
        else:  # Demo mode
            print("⚠️  Running in demo mode with limited functionality")
            # Use a mock provider for demo
            provider = OpenAIProvider("demo-key")  # This will fail gracefully
    
        # Initialize system
        print("\\n🔧 Initializing LLM-powered FIBO-LightRAG system...")
        system = FiboLightRAGLLMSystem(provider)
        
        # Show initial statistics
        stats = system.get_statistics()
        print(f"\\n📊 System Statistics:")
        print(f"   FIBO Classes: {stats['fibo_ontology']['classes']}")
        print(f"   Vector Dimension: {stats['vector_store']['embedding_dimension']}")
        
        # Load sample documents
        print("\\n📄 Loading sample financial documents...")
        documents = create_sample_financial_documents()
        
        for doc in documents:
            print(f"   Processing {doc['id']}...")
            success = system.add_document(
                content=doc['content'],
                doc_id=doc['id'],
                metadata=doc['metadata']
            )
            status = "✅" if success else "❌"
            print(f"   {status} {doc['id']}")
        
        # Updated statistics
        stats = system.get_statistics()
        print(f"\\n📊 Updated Statistics:")
        print(f"   Documents: {stats['vector_store']['total_documents']}")
        print(f"   Graph Nodes: {stats['knowledge_graph']['num_nodes']}")
        print(f"   Graph Edges: {stats['knowledge_graph']['num_edges']}")
        
        # Sample queries
        print("\\n🔍 Running sample queries with LLM analysis...")
        
        queries = [
            "What is JPMorgan Chase's financial performance?",
            "Tell me about Goldman Sachs partnerships",
            "How are banks using Microsoft cloud services?",
            "Compare the revenue of financial institutions"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\\n🔍 Query {i}: {query}")
            
            response = system.query(query)
            print(f"   📈 Results: {len(response.results)} found")
            print(f"   🤖 LLM Analysis: {response.llm_analysis[:200]}...")
            
            if response.results:
                top_result = response.results[0]
                print(f"   📄 Top Result Score: {top_result.score:.3f}")
                print(f"   💡 Summary: {top_result.llm_summary[:150]}...")
        
        print("\\n🎉 LLM-powered demo completed successfully!")
        print("\\nThe system demonstrated:")
        print("✓ Real LLM-based entity extraction")
        print("✓ Neural embedding similarity search")
        print("✓ LLM relationship inference")
        print("✓ Intelligent query analysis")
        print("✓ FIBO ontology integration")
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for LLM-powered FIBO-LightRAG."""
    print("🤖 FIBO-LightRAG: LLM-Powered Financial Knowledge Graph System")
    print("=" * 70)
    print()
    print("This version uses real LLMs and embeddings for:")
    print("• Entity extraction with FIBO ontology mapping")
    print("• Relationship inference using natural language understanding")
    print("• Neural embeddings for semantic similarity")
    print("• Intelligent query processing and analysis")
    print()
    
    run_llm_demo()

if __name__ == "__main__":
    main()