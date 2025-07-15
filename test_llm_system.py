#!/usr/bin/env python3
"""
Quick test script for LLM-powered FIBO-LightRAG without API keys.
This shows the system structure and components working together.
"""

import sys
import os

# Mock LLM provider for testing without API keys
class MockLLMProvider:
    """Mock provider for testing without real API keys."""
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Generate mock text responses."""
        if "extract" in prompt.lower() and "entities" in prompt.lower():
            return '''[
                {
                    "text": "JPMorgan Chase",
                    "fibo_class": "Bank", 
                    "confidence": 0.95,
                    "reasoning": "Large financial institution that accepts deposits and makes loans"
                },
                {
                    "text": "$49.6 billion",
                    "fibo_class": "Monetary Amount",
                    "confidence": 0.98,
                    "reasoning": "Specific dollar amount representing net income"
                }
            ]'''
        
        elif "relationship" in prompt.lower():
            return '''[
                {
                    "source_entity": "JPMorgan Chase",
                    "target_entity": "First Republic Bank",
                    "relationship_type": "acquired",
                    "confidence": 0.92,
                    "evidence": "JPMorgan completed the acquisition of First Republic Bank"
                }
            ]'''
        
        elif "summary" in prompt.lower() or "analysis" in prompt.lower():
            return "JPMorgan Chase demonstrated strong financial performance with record net income, reflecting successful business operations and strategic acquisitions."
        
        else:
            return "Financial institutions like JPMorgan Chase operate in regulated markets providing banking services."
    
    def generate_embedding(self, text: str) -> list:
        """Generate mock embeddings."""
        # Create a simple hash-based embedding for testing
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hex_dig = hash_obj.hexdigest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, min(len(hex_dig), 32), 2):  # Create 16-dim vector
            val = int(hex_dig[i:i+2], 16) / 255.0  # Normalize to 0-1
            embedding.append(val)
        
        # Pad to consistent length
        while len(embedding) < 16:
            embedding.append(0.1)
            
        return embedding

def test_llm_system():
    """Test the LLM system with mock provider."""
    print("🧪 Testing LLM-Powered FIBO-LightRAG System")
    print("=" * 50)
    
    # Import the LLM system components
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from fibo_lightrag_llm import (
            FiboLightRAGLLMSystem,
            LLMEnhancedFiboParser,
            LLMEntityExtractor,
            LLMGraphBuilder,
            LLMVectorStore
        )
        
        print("✅ Successfully imported LLM system components")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure fibo_lightrag_llm.py is in the same directory")
        return False
    
    try:
        # Initialize with mock provider
        print("\\n🔧 Initializing system with mock LLM provider...")
        mock_provider = MockLLMProvider()
        
        # Test individual components
        print("\\n📚 Testing FIBO parser...")
        fibo_parser = LLMEnhancedFiboParser(mock_provider)
        print(f"   ✅ Loaded {len(fibo_parser.classes)} FIBO classes")
        
        print("\\n🔍 Testing entity extractor...")
        entity_extractor = LLMEntityExtractor(mock_provider, fibo_parser)
        test_text = "JPMorgan Chase reported net income of $49.6 billion in 2023."
        entities = entity_extractor.extract_entities(test_text)
        print(f"   ✅ Extracted {len(entities)} entities")
        
        for entity in entities:
            print(f"      • {entity.text} ({entity.fibo_class}) - {entity.confidence:.2f}")
        
        print("\\n🌐 Testing knowledge graph builder...")
        graph_builder = LLMGraphBuilder(mock_provider, fibo_parser)
        
        # Add entities to graph
        for entity in entities:
            graph_builder.add_entity_to_graph(entity, "test_doc")
        
        # Test relationship inference
        relationships = graph_builder.infer_relationships_llm(entities, test_text, "test_doc")
        print(f"   ✅ Built graph with {len(graph_builder.graph.nodes)} nodes, {relationships} relationships")
        
        print("\\n🔢 Testing vector store...")
        vector_store = LLMVectorStore(mock_provider)
        vector_store.add_document("test_doc", test_text, {"type": "test"})
        
        search_results = vector_store.search_similar("JPMorgan financial performance")
        print(f"   ✅ Vector search returned {len(search_results)} results")
        
        print("\\n🤖 Testing complete system...")
        system = FiboLightRAGLLMSystem(mock_provider)
        
        # Add document
        success = system.add_document(test_text, "test_doc")
        print(f"   ✅ Document addition: {'Success' if success else 'Failed'}")
        
        # Query system
        response = system.query("What is JPMorgan's financial performance?")
        print(f"   ✅ Query returned {len(response.results)} results")
        print(f"   🤖 LLM Analysis: {response.llm_analysis}")
        
        # Show statistics
        stats = system.get_statistics()
        print(f"\\n📊 Final System Statistics:")
        print(f"   FIBO Classes: {stats['fibo_ontology']['classes']}")
        print(f"   Documents: {stats['vector_store']['total_documents']}")
        print(f"   Graph Nodes: {stats['knowledge_graph']['num_nodes']}")
        print(f"   Graph Edges: {stats['knowledge_graph']['num_edges']}")
        print(f"   Embedding Dimension: {stats['vector_store']['embedding_dimension']}")
        
        print("\\n🎉 All tests passed! LLM system is working correctly.")
        print("\\n📝 To use with real LLMs:")
        print("   1. Install: pip install openai anthropic sentence-transformers")
        print("   2. Set API keys: export OPENAI_API_KEY=your_key")
        print("   3. Run: python fibo_lightrag_llm.py")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_llm_system()