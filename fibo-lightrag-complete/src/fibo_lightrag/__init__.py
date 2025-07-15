"""
FIBO-LightRAG: Financial Knowledge Graph System

A system that combines LightRAG concepts with Financial Industry Business Ontology (FIBO)
for advanced financial document analysis using LLMs and embeddings.
"""

__version__ = "1.0.0"
__author__ = "FIBO-LightRAG Team"

from .integration.fibo_lightrag_system import FiboLightRAGSystem
from .integration.config import FiboLightRAGConfig

__all__ = [
    "FiboLightRAGSystem",
    "FiboLightRAGConfig"
]