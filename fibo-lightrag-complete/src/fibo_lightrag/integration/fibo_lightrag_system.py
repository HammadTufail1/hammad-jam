"""
Main FIBO-LightRAG system integration and orchestration.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..fibo.parser import FiboParser
from ..lightrag.entity_extractor import FiboEntityExtractor
from ..graph.graph_builder import FiboGraphBuilder
from ..retrieval.document_processor import FinancialDocumentProcessor
from ..retrieval.vector_store import FinancialVectorStore
from ..retrieval.retrieval_engine import FiboRetrievalEngine, RetrievalResult
from ..retrieval.query_processor import FinancialQueryProcessor
from .config import FiboLightRAGConfig

logger = logging.getLogger(__name__)

@dataclass
class SystemResponse:
    """Response from the FIBO-LightRAG system."""
    results: List[RetrievalResult]
    query_intent: Optional[Any]
    processing_time: float
    metadata: Dict[str, Any]

class FiboLightRAGSystem:
    """Main FIBO-LightRAG system orchestrating all components."""
    
    def __init__(self, config: Optional[FiboLightRAGConfig] = None):
        self.config = config or FiboLightRAGConfig()
        self.config.setup_logging()
        
        # Initialize components
        self.fibo_parser = None
        self.entity_extractor = None
        self.graph_builder = None
        self.document_processor = None
        self.vector_store = None
        self.retrieval_engine = None
        self.query_processor = None
        
        self._initialized = False
        
        logger.info("FIBO-LightRAG system created")
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("Initializing FIBO-LightRAG system...")
            
            # Initialize FIBO parser
            self._initialize_fibo_parser()
            
            # Initialize entity extractor
            self._initialize_entity_extractor()
            
            # Initialize graph builder
            self._initialize_graph_builder()
            
            # Initialize document processor
            self._initialize_document_processor()
            
            # Initialize vector store
            self._initialize_vector_store()
            
            # Initialize query processor
            self._initialize_query_processor()
            
            # Initialize retrieval engine (must be last)
            self._initialize_retrieval_engine()
            
            self._initialized = True
            logger.info("FIBO-LightRAG system successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FIBO-LightRAG system: {e}")
            return False
    
    def _initialize_fibo_parser(self) -> None:
        """Initialize FIBO ontology parser."""
        self.fibo_parser = FiboParser()
        
        # Try to load parsed ontology data
        if os.path.exists(self.config.fibo_ontology_path):
            success = self.fibo_parser.load_parsed_data(self.config.fibo_ontology_path)
            if success:
                logger.info(f"Loaded FIBO ontology from {self.config.fibo_ontology_path}")
            else:
                logger.warning("Failed to load parsed FIBO data, will use empty ontology")
        else:
            logger.warning(f"FIBO ontology file not found at {self.config.fibo_ontology_path}")
            # Create sample data for demonstration
            self._create_sample_fibo_data()
    
    def _create_sample_fibo_data(self) -> None:
        """Create sample FIBO data for demonstration purposes."""
        logger.info("Creating sample FIBO data for demonstration")
        
        # Create sample classes
        from ..fibo.parser import FiboClass, FiboProperty, FiboRelationship
        
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
            )
        }
        
        self.fibo_parser.classes = sample_classes
        
        sample_properties = {
            "https://spec.edmcouncil.org/fibo/ontology/hasName": FiboProperty(
                uri="https://spec.edmcouncil.org/fibo/ontology/hasName",
                label="hasName",
                definition="The name of an entity",
                domain=["https://spec.edmcouncil.org/fibo/ontology/Organization"],
                range=["string"]
            )
        }
        
        self.fibo_parser.properties = sample_properties
        
        sample_relationships = [
            FiboRelationship(
                subject="https://spec.edmcouncil.org/fibo/ontology/Bank",
                predicate="https://spec.edmcouncil.org/fibo/ontology/subClassOf",
                object="https://spec.edmcouncil.org/fibo/ontology/Organization",
                relationship_type="subclass"
            )
        ]
        
        self.fibo_parser.relationships = sample_relationships
        
        logger.info(f"Created sample FIBO data: {len(sample_classes)} classes, {len(sample_properties)} properties")
    
    def _initialize_entity_extractor(self) -> None:
        """Initialize entity extractor."""
        self.entity_extractor = FiboEntityExtractor(self.fibo_parser)
        logger.info("Entity extractor initialized")
    
    def _initialize_graph_builder(self) -> None:
        """Initialize knowledge graph builder."""
        self.graph_builder = FiboGraphBuilder(self.fibo_parser)
        logger.info("Knowledge graph builder initialized")
    
    def _initialize_document_processor(self) -> None:
        """Initialize document processor."""
        vector_config = self.config.get_vector_config()
        self.document_processor = FinancialDocumentProcessor(
            chunk_size=vector_config['chunk_size'],
            chunk_overlap=vector_config['chunk_overlap']
        )
        logger.info("Document processor initialized")
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store."""
        vector_config = self.config.get_vector_config()
        self.vector_store = FinancialVectorStore(
            dimension=vector_config['dimension']
        )
        logger.info("Vector store initialized")
    
    def _initialize_query_processor(self) -> None:
        """Initialize query processor."""
        self.query_processor = FinancialQueryProcessor()
        logger.info("Query processor initialized")
    
    def _initialize_retrieval_engine(self) -> None:
        """Initialize retrieval engine."""
        self.retrieval_engine = FiboRetrievalEngine(
            vector_store=self.vector_store,
            graph_builder=self.graph_builder,
            entity_extractor=self.entity_extractor,
            query_processor=self.query_processor
        )
        
        # Update retrieval engine configuration
        retrieval_config = self.config.get_retrieval_config()
        self.retrieval_engine.config.update(retrieval_config)
        
        logger.info("Retrieval engine initialized")
    
    def add_document(self, content: str, doc_id: Optional[str] = None, 
                    metadata: Optional[Dict] = None) -> bool:
        """Add a document to the system."""
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if doc_id is None:
            doc_id = f"doc_{hash(content[:100])}"
        
        try:
            # Process document into chunks
            chunks, metrics = self.document_processor.process_document(content, doc_id)
            
            # Add document to retrieval engine (handles both vector and graph)
            success = self.retrieval_engine.add_document(doc_id, content, metadata)
            
            if success:
                logger.info(f"Successfully added document {doc_id}: {len(chunks)} chunks, {len(metrics)} metrics")
            else:
                logger.error(f"Failed to add document {doc_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    def query(self, query: str, method: Optional[str] = None) -> SystemResponse:
        """Query the system for information."""
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        import time
        start_time = time.time()
        
        try:
            # Process query to understand intent
            query_intent = self.query_processor.process_query(query)
            
            # Determine retrieval method
            if method is None:
                method = self.config.retrieval_method
            
            # Retrieve results
            results = self.retrieval_engine.retrieve(query, method)
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'query': query,
                'method': method,
                'num_results': len(results),
                'processing_time': processing_time,
                'system_stats': self.get_statistics()
            }
            
            response = SystemResponse(
                results=results,
                query_intent=query_intent,
                processing_time=processing_time,
                metadata=metadata
            )
            
            logger.info(f"Query processed: '{query}' -> {len(results)} results in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            
            # Return empty response on error
            return SystemResponse(
                results=[],
                query_intent=None,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        stats = {
            'status': 'initialized',
            'fibo_ontology': {
                'classes': len(self.fibo_parser.classes),
                'properties': len(self.fibo_parser.properties),
                'relationships': len(self.fibo_parser.relationships)
            }
        }
        
        if self.retrieval_engine:
            stats.update(self.retrieval_engine.get_statistics())
        
        return stats