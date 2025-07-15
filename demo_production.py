#!/usr/bin/env python3
"""
Production Demo: FIBO-LightRAG with Real LLMs
==============================================

This demo shows how to use the FIBO-LightRAG system with real LLM providers
for analyzing financial documents and reports.

Setup:
1. Install dependencies: pip install openai anthropic google-generativeai sentence-transformers
2. Set API keys as environment variables
3. Run: python demo_production.py

Author: FIBO-LightRAG Team
"""

import os
import sys
from typing import Dict, Any

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import openai
    except ImportError:
        missing.append("openai")
    
    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")
    
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"📦 Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_api_keys():
    """Check if API keys are configured."""
    keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'Google': os.getenv('GOOGLE_API_KEY')
    }
    
    available = []
    for provider, key in keys.items():
        if key:
            available.append(provider)
            print(f"✅ {provider} API key configured")
        else:
            print(f"⚠️  {provider} API key not found (set {provider.upper().replace('GOOGLE', 'GOOGLE')}_API_KEY)")
    
    if not available:
        print("\n❌ No API keys configured. Using mock provider for demo.")
        return None
    
    return available[0].lower()  # Return first available provider

def create_sample_financial_report():
    """Create a sample financial report for demonstration."""
    return """
    JPMORGAN CHASE & CO. - 2023 ANNUAL REPORT EXCERPT
    
    Financial Highlights:
    - Net income: $49.6 billion in 2023, up from $37.7 billion in 2022
    - Return on equity: 15% in 2023
    - Book value per share: $95.87
    - Total assets: $3.7 trillion
    
    Business Overview:
    JPMorgan Chase & Co. is a leading global financial services firm with operations 
    in more than 60 countries. The firm is a leader in investment banking, consumer 
    and small business banking, commercial banking, financial transaction processing 
    and asset management.
    
    Key Acquisitions:
    - Acquired First Republic Bank in May 2023, adding $173 billion in loans
    - Strategic partnership with Apple for Apple Card services
    - Expansion into wealth management through new advisory services
    
    Risk Management:
    The firm maintains strong capital ratios with a CET1 ratio of 15.0%.
    Credit loss provisions totaled $1.4 billion in 2023.
    
    ESG Commitments:
    - $2.5 trillion sustainable financing commitment through 2030
    - Net zero operational emissions by 2030
    - Supporting 40,000 jobs in underinvested communities
    """

def run_production_demo():
    """Run the production demo with real or mock LLM provider."""
    
    print("🚀 FIBO-LightRAG Production Demo")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check API keys
    provider = check_api_keys()
    
    print(f"\n🤖 Using LLM Provider: {provider or 'Mock (for demo)'}")
    print("-" * 30)
    
    # Import and setup system
    try:
        if provider:
            # Import the real system
            from fibo_lightrag_llm import FiboLightRAGLLMSystem, OpenAIProvider, AnthropicProvider
            
            # Initialize with real provider
            if provider == 'openai':
                llm_provider = OpenAIProvider()
            elif provider == 'anthropic':
                llm_provider = AnthropicProvider()
            else:
                print("⚠️ Unsupported provider, using mock")
                from test_llm_system import MockLLMProvider
                llm_provider = MockLLMProvider()
        else:
            # Use mock provider
            from test_llm_system import MockLLMProvider
            from fibo_lightrag_llm import FiboLightRAGLLMSystem
            llm_provider = MockLLMProvider()
        
        # Initialize system
        print("🔧 Initializing FIBO-LightRAG system...")
        system = FiboLightRAGLLMSystem(llm_provider=llm_provider)
        
        # Add sample document
        print("📄 Adding sample financial report...")
        sample_report = create_sample_financial_report()
        system.add_document("jpmorgan_2023_report", sample_report)
        
        # Demo queries
        queries = [
            "What was JPMorgan's net income in 2023?",
            "Tell me about JPMorgan's acquisitions",
            "What are JPMorgan's ESG commitments?",
            "How is JPMorgan managing risk?",
            "What is JPMorgan's return on equity?"
        ]
        
        print("\n🔍 Running Demo Queries:")
        print("-" * 25)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            results = system.query(query)
            
            print(f"   📊 Found {len(results['retrieved_chunks'])} relevant chunks")
            if results['retrieved_chunks']:
                # Show first result excerpt
                first_chunk = results['retrieved_chunks'][0]
                excerpt = first_chunk['text'][:200] + "..." if len(first_chunk['text']) > 200 else first_chunk['text']
                print(f"   📖 Excerpt: {excerpt}")
            
            if results.get('llm_analysis'):
                analysis = results['llm_analysis'][:300] + "..." if len(results['llm_analysis']) > 300 else results['llm_analysis']
                print(f"   🤖 Analysis: {analysis}")
        
        # System statistics
        stats = system.get_statistics()
        print(f"\n📈 System Statistics:")
        print(f"   • Documents: {stats.get('documents', 0)}")
        print(f"   • Graph Nodes: {stats.get('graph_nodes', 0)}")
        print(f"   • Graph Edges: {stats.get('graph_edges', 0)}")
        print(f"   • FIBO Classes: {stats.get('fibo_classes', 0)}")
        
        print("\n✅ Demo completed successfully!")
        
        # Next steps
        print(f"\n🎯 Next Steps:")
        print(f"   1. Set up your API keys for production use")
        print(f"   2. Add your financial documents using system.add_document()")
        print(f"   3. Query for insights using system.query()")
        print(f"   4. Export graphs using system.export_graph()")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_production_demo()