#!/usr/bin/env python3
"""
Demo script for FIBO-LightRAG system.
Demonstrates the core functionality with sample financial documents.
"""

import os
import sys
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fibo_lightrag.integration.fibo_lightrag_system import FiboLightRAGSystem
from fibo_lightrag.integration.config import FiboLightRAGConfig

def setup_demo_logging():
    """Setup logging for demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_sample_documents():
    """Create sample financial documents for demonstration."""
    documents = [
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
            
            Key financial ratios:
            - Return on Equity (ROE): 14.2%
            - Return on Assets (ROA): 1.8%
            - Tier 1 Capital Ratio: 12.5%
            - Net Interest Margin: 3.4%
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
            up 35% year-over-year. Software licensing revenue was $2.8 billion, 
            growing 18% from the previous year.
            
            TechCorp announced a new partnership with Global Financial Services Inc.
            to provide enterprise software solutions. The company also acquired 
            DataAnalytics Pro for $150 million to enhance its AI capabilities.
            
            Operating margin improved to 28.5% from 26.1% in the prior year quarter.
            The company maintains a strong balance sheet with $12 billion in cash 
            and short-term investments.
            """,
            'metadata': {
                'document_type': 'earnings_transcript',
                'quarter': 'Q4',
                'year': 2023,
                'company': 'TechCorp'
            }
        },
        {
            'id': 'market_analysis_2024',
            'content': """
            Financial Market Analysis - Banking Sector Outlook 2024
            
            The banking sector is expected to face mixed conditions in 2024. 
            Interest rate stabilization should benefit net interest margins, 
            but credit concerns may impact loan growth.
            
            Large banks like First National Bank and Metropolitan Trust are 
            well-positioned due to their diversified revenue streams and 
            strong capital positions. Regional banks may face more pressure 
            from commercial real estate exposures.
            
            Digital transformation continues to drive efficiency gains across 
            the industry. Banks investing in fintech partnerships and AI-driven 
            risk management systems are showing improved operational metrics.
            
            Key trends to watch:
            - Regulatory changes in capital requirements
            - Competition from fintech companies
            - Adoption of central bank digital currencies (CBDCs)
            - ESG reporting requirements
            """,
            'metadata': {
                'document_type': 'market_analysis',
                'year': 2024,
                'sector': 'banking'
            }
        }
    ]
    
    return documents

def run_demo():
    """Run the FIBO-LightRAG system demo."""
    print("🏦 FIBO-LightRAG System Demo")
    print("=" * 50)
    
    # Setup logging
    setup_demo_logging()
    
    # Create system with default configuration
    print("\n1. Initializing FIBO-LightRAG system...")
    config = FiboLightRAGConfig()
    system = FiboLightRAGSystem(config)
    
    # Initialize the system
    if not system.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully")
    
    # Show initial statistics
    stats = system.get_statistics()
    print(f"\n📊 Initial System Statistics:")
    print(f"   FIBO Classes: {stats['fibo_ontology']['classes']}")
    print(f"   FIBO Properties: {stats['fibo_ontology']['properties']}")
    print(f"   FIBO Relationships: {stats['fibo_ontology']['relationships']}")
    
    # Add sample documents
    print("\n2. Adding sample financial documents...")
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
    print(f"\n📊 Updated System Statistics:")
    print(f"   Documents in Vector Store: {stats['vector_store']['total_documents']}")
    print(f"   Knowledge Graph Nodes: {stats['knowledge_graph']['num_nodes']}")
    print(f"   Knowledge Graph Edges: {stats['knowledge_graph']['num_edges']}")
    
    # Run sample queries
    print("\n3. Running sample queries...")
    
    queries = [
        "What is First National Bank's revenue?",
        "Tell me about TechCorp's partnerships",
        "How did banking sector perform in 2023?",
        "What are the key financial ratios for banks?",
        "Show me companies with revenue growth"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        # Test different retrieval methods
        for method in ['vector', 'graph', 'hybrid']:
            response = system.query(query, method=method)
            
            print(f"   📈 {method.title()} Results: {len(response.results)} found")
            
            # Show top result if available
            if response.results:
                top_result = response.results[0]
                content_preview = top_result.content[:100] + "..." if len(top_result.content) > 100 else top_result.content
                print(f"      Score: {top_result.score:.3f}")
                print(f"      Preview: {content_preview}")
    
    # Show final system analysis
    print("\n4. System Analysis Summary")
    print("-" * 30)
    
    final_stats = system.get_statistics()
    
    print(f"Knowledge Graph Analysis:")
    if 'knowledge_graph' in final_stats:
        kg_stats = final_stats['knowledge_graph']
        print(f"  • Total Nodes: {kg_stats['num_nodes']}")
        print(f"  • Total Edges: {kg_stats['num_edges']}")
        
        if 'node_types' in kg_stats:
            print(f"  • Node Types: {list(kg_stats['node_types'].keys())}")
        
        if 'relationship_types' in kg_stats:
            print(f"  • Relationship Types: {kg_stats['relationship_types']}")
    
    print(f"\nVector Store Analysis:")
    if 'vector_store' in final_stats:
        vs_stats = final_stats['vector_store']
        print(f"  • Documents: {vs_stats['total_documents']}")
        print(f"  • Average Length: {vs_stats.get('average_content_length', 0):.0f} chars")
        print(f"  • Vector Dimension: {vs_stats['vector_dimension']}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe FIBO-LightRAG system demonstrated:")
    print("✓ FIBO ontology integration")
    print("✓ Financial entity extraction")
    print("✓ Knowledge graph construction")
    print("✓ Multi-modal retrieval (vector + graph)")
    print("✓ Query processing and understanding")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()