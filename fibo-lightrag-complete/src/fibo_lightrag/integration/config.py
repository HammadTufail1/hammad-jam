"""
Configuration management for FIBO-LightRAG system.
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class FiboLightRAGConfig:
    """Configuration for FIBO-LightRAG system."""
    
    # FIBO Ontology settings
    fibo_ontology_path: str = "data/fibo_parsed.json"
    fibo_download_urls: Dict[str, str] = None
    
    # Vector store settings
    vector_dimension: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Entity extraction settings
    entity_confidence_threshold: float = 0.5
    entity_context_window: int = 50
    enable_regex_extraction: bool = True
    enable_contextual_extraction: bool = True
    
    # Knowledge graph settings
    enable_inference: bool = True
    max_inference_depth: int = 2
    relationship_confidence_threshold: float = 0.3
    
    # Retrieval settings
    retrieval_method: str = "hybrid"  # vector, graph, hybrid
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    max_results: int = 10
    min_similarity: float = 0.1
    max_graph_depth: int = 2
    
    # Query processing settings
    enable_query_expansion: bool = True
    enable_query_refinement: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    enable_debug_logging: bool = False
    
    # Performance settings
    max_concurrent_processes: int = 4
    cache_embeddings: bool = True
    cache_graph_queries: bool = True
    
    def __post_init__(self):
        """Initialize default FIBO download URLs if not provided."""
        if self.fibo_download_urls is None:
            self.fibo_download_urls = {
                "organizations": "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/",
                "agents": "https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/Agents/",
                "business_registries": "https://spec.edmcouncil.org/fibo/ontology/BE/GovernmentEntities/GovernmentEntities/",
                "financial_products": "https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/"
            }
    
    def validate(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate vector settings
        if self.vector_dimension <= 0:
            errors.append("vector_dimension must be positive")
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be between 0 and chunk_size")
        
        # Validate thresholds
        if not 0 <= self.entity_confidence_threshold <= 1:
            errors.append("entity_confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.relationship_confidence_threshold <= 1:
            errors.append("relationship_confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.min_similarity <= 1:
            errors.append("min_similarity must be between 0 and 1")
        
        # Validate weights
        if not 0 <= self.vector_weight <= 1:
            errors.append("vector_weight must be between 0 and 1")
        
        if not 0 <= self.graph_weight <= 1:
            errors.append("graph_weight must be between 0 and 1")
        
        # Validate retrieval method
        valid_methods = ["vector", "graph", "hybrid"]
        if self.retrieval_method not in valid_methods:
            errors.append(f"retrieval_method must be one of {valid_methods}")
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            errors.append(f"log_level must be one of {valid_levels}")
        
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False
        
        return True
    
    def save_to_file(self, filepath: str) -> bool:
        """Save configuration to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['FiboLightRAGConfig']:
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            config = cls(**data)
            if config.validate():
                logger.info(f"Configuration loaded from {filepath}")
                return config
            else:
                logger.error("Loaded configuration is invalid")
                return None
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def update(self, **kwargs) -> bool:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        return self.validate()
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get configuration specific to retrieval engine."""
        return {
            'vector_weight': self.vector_weight,
            'graph_weight': self.graph_weight,
            'max_results': self.max_results,
            'min_similarity': self.min_similarity,
            'confidence_threshold': self.relationship_confidence_threshold,
            'max_graph_depth': self.max_graph_depth,
            'enable_relationship_inference': self.enable_inference
        }
    
    def get_entity_extraction_config(self) -> Dict[str, Any]:
        """Get configuration specific to entity extraction."""
        return {
            'confidence_threshold': self.entity_confidence_threshold,
            'context_window': self.entity_context_window,
            'enable_regex': self.enable_regex_extraction,
            'enable_contextual': self.enable_contextual_extraction
        }
    
    def get_graph_config(self) -> Dict[str, Any]:
        """Get configuration specific to knowledge graph."""
        return {
            'enable_inference': self.enable_inference,
            'max_inference_depth': self.max_inference_depth,
            'relationship_confidence_threshold': self.relationship_confidence_threshold
        }
    
    def get_vector_config(self) -> Dict[str, Any]:
        """Get configuration specific to vector store."""
        return {
            'dimension': self.vector_dimension,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'cache_embeddings': self.cache_embeddings
        }
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        level = getattr(logging, self.log_level)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        # Enable debug logging for specific modules if requested
        if self.enable_debug_logging:
            debug_loggers = [
                'fibo_lightrag.fibo',
                'fibo_lightrag.lightrag',
                'fibo_lightrag.graph',
                'fibo_lightrag.retrieval'
            ]
            
            for logger_name in debug_loggers:
                logging.getLogger(logger_name).setLevel(logging.DEBUG)
        
        logger.info(f"Logging configured at level: {self.log_level}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"FiboLightRAGConfig(method={self.retrieval_method}, vector_dim={self.vector_dimension}, chunk_size={self.chunk_size})"