"""
Vector storage and similarity search for financial documents.
Uses a simple vector implementation to avoid heavy ML dependencies.
"""

import json
import logging
import math
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SimpleVector:
    """Simple vector implementation for embeddings."""
    data: List[float]
    
    def __post_init__(self):
        if not isinstance(self.data, list):
            raise ValueError("Vector data must be a list of floats")
    
    @property
    def dimension(self) -> int:
        return len(self.data)
    
    def dot_product(self, other: 'SimpleVector') -> float:
        """Calculate dot product with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(sum(x * x for x in self.data))
    
    def cosine_similarity(self, other: 'SimpleVector') -> float:
        """Calculate cosine similarity with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have same dimension")
        
        dot_prod = self.dot_product(other)
        mag_product = self.magnitude() * other.magnitude()
        
        if mag_product == 0:
            return 0.0
        
        return dot_prod / mag_product
    
    def normalize(self) -> 'SimpleVector':
        """Return normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            return SimpleVector(self.data.copy())
        
        return SimpleVector([x / mag for x in self.data])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimpleVector':
        """Create vector from dictionary."""
        return cls(data['data'])

@dataclass
class VectorDocument:
    """Document with vector embedding."""
    id: str
    content: str
    vector: SimpleVector
    metadata: Dict[str, Any]

class FinancialVectorStore:
    """Vector store for financial document embeddings."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.documents: Dict[str, VectorDocument] = {}
        self.index_dirty = True
        self._build_simple_vocab()
    
    def _build_simple_vocab(self):
        """Build a simple vocabulary for text-to-vector conversion."""
        # Financial terms vocabulary
        self.financial_vocab = [
            # Basic financial terms
            'revenue', 'profit', 'loss', 'income', 'expense', 'cost', 'margin',
            'asset', 'liability', 'equity', 'debt', 'cash', 'investment', 'capital',
            'earnings', 'dividend', 'share', 'stock', 'bond', 'security', 'fund',
            'bank', 'loan', 'credit', 'deposit', 'interest', 'rate', 'yield',
            'growth', 'decline', 'increase', 'decrease', 'performance', 'return',
            'ratio', 'percentage', 'quarter', 'annual', 'fiscal', 'financial',
            'market', 'economy', 'industry', 'sector', 'company', 'corporation',
            'business', 'operation', 'strategy', 'management', 'board', 'executive',
            'risk', 'compliance', 'regulation', 'audit', 'report', 'statement',
            'balance', 'sheet', 'flow', 'budget', 'forecast', 'analysis',
            
            # Common business terms  
            'customer', 'client', 'partner', 'supplier', 'vendor', 'contract',
            'agreement', 'merger', 'acquisition', 'subsidiary', 'division',
            'product', 'service', 'offering', 'solution', 'technology', 'innovation',
            'sales', 'marketing', 'operations', 'human', 'resources', 'legal',
            
            # Time and dates
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'year', 'month', 'week', 'day', 'quarter', 'semester',
            
            # Numbers and quantities (representations)
            'million', 'billion', 'trillion', 'thousand', 'hundred',
            'first', 'second', 'third', 'fourth', 'fifth',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            
            # Common adjectives and descriptors
            'high', 'low', 'strong', 'weak', 'positive', 'negative', 'stable', 'volatile',
            'significant', 'minor', 'major', 'critical', 'important', 'key', 'primary',
            'secondary', 'new', 'old', 'current', 'previous', 'future', 'potential',
            'actual', 'estimated', 'projected', 'expected', 'reported', 'disclosed'
        ]
        
        # Pad vocabulary to ensure minimum dimension
        while len(self.financial_vocab) < self.dimension:
            self.financial_vocab.extend(['generic', 'term', 'word', 'text', 'document'])
        
        self.financial_vocab = self.financial_vocab[:self.dimension]
    
    def create_dummy_embedding(self, text: str) -> SimpleVector:
        """Create a simple embedding based on financial term frequency."""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Initialize vector with zeros
        vector_data = [0.0] * self.dimension
        
        # Term frequency approach
        for i, term in enumerate(self.financial_vocab):
            if i >= self.dimension:
                break
                
            # Count occurrences of term in text
            term_count = sum(1 for word in words if term in word)
            
            if term_count > 0:
                # Use log frequency with some normalization
                vector_data[i] = math.log(1 + term_count)
        
        # Add some variation based on text characteristics
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        
        for i in range(min(20, self.dimension)):  # First 20 dimensions for text-specific features
            # Add small random-like values based on hash
            vector_data[i] += (((text_hash >> i) & 1) - 0.5) * 0.1
        
        # Normalize to prevent extremely large values
        max_val = max(abs(x) for x in vector_data) if vector_data else 1
        if max_val > 0:
            vector_data = [x / max_val for x in vector_data]
        
        return SimpleVector(vector_data)
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document to the vector store."""
        try:
            # Create embedding
            vector = self.create_dummy_embedding(content)
            
            # Create vector document
            doc = VectorDocument(
                id=doc_id,
                content=content,
                vector=vector,
                metadata=metadata or {}
            )
            
            self.documents[doc_id] = doc
            self.index_dirty = True
            
            logger.debug(f"Added document {doc_id} to vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the vector store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.index_dirty = True
            logger.debug(f"Removed document {doc_id} from vector store")
            return True
        return False
    
    def search_similar(self, query: str, top_k: int = 10, 
                      min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar documents using cosine similarity."""
        if not self.documents:
            return []
        
        # Create query vector
        query_vector = self.create_dummy_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc in self.documents.items():
            similarity = query_vector.cosine_similarity(doc.vector)
            
            if similarity >= min_similarity:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def search_similar_to_document(self, doc_id: str, top_k: int = 10,
                                 exclude_self: bool = True) -> List[Tuple[str, float]]:
        """Find documents similar to a given document."""
        if doc_id not in self.documents:
            return []
        
        target_doc = self.documents[doc_id]
        similarities = []
        
        for other_id, other_doc in self.documents.items():
            if exclude_self and other_id == doc_id:
                continue
            
            similarity = target_doc.vector.cosine_similarity(other_doc.vector)
            similarities.append((other_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> Dict[str, VectorDocument]:
        """Get all documents."""
        return self.documents.copy()
    
    def filter_documents(self, filter_func) -> List[VectorDocument]:
        """Filter documents using a custom function."""
        return [doc for doc in self.documents.values() if filter_func(doc)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.documents:
            return {
                'total_documents': 0,
                'vector_dimension': self.dimension,
                'average_content_length': 0,
                'metadata_keys': []
            }
        
        total_length = sum(len(doc.content) for doc in self.documents.values())
        avg_length = total_length / len(self.documents)
        
        # Collect all metadata keys
        all_metadata_keys = set()
        for doc in self.documents.values():
            all_metadata_keys.update(doc.metadata.keys())
        
        return {
            'total_documents': len(self.documents),
            'vector_dimension': self.dimension,
            'average_content_length': avg_length,
            'total_content_length': total_length,
            'metadata_keys': list(all_metadata_keys)
        }
    
    def save_to_file(self, filepath: str) -> bool:
        """Save vector store to JSON file."""
        try:
            data = {
                'dimension': self.dimension,
                'documents': {
                    doc_id: {
                        'id': doc.id,
                        'content': doc.content,
                        'vector': doc.vector.to_dict(),
                        'metadata': doc.metadata
                    }
                    for doc_id, doc in self.documents.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved vector store to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load vector store from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.dimension = data['dimension']
            self.documents = {}
            
            for doc_id, doc_data in data['documents'].items():
                vector = SimpleVector.from_dict(doc_data['vector'])
                doc = VectorDocument(
                    id=doc_data['id'],
                    content=doc_data['content'],
                    vector=vector,
                    metadata=doc_data['metadata']
                )
                self.documents[doc_id] = doc
            
            self.index_dirty = True
            logger.info(f"Loaded vector store from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.documents = {}
        self.index_dirty = True
        logger.info("Cleared vector store")