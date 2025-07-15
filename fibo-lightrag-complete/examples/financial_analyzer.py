#!/usr/bin/env python3
"""
Financial Analyzer Example using FIBO-LightRAG

This example demonstrates how to use FIBO-LightRAG for comprehensive
financial document analysis and reporting.
"""

import os
import sys
import json
from typing import Dict, List, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fibo_lightrag.integration.fibo_lightrag_system import FiboLightRAGSystem
from fibo_lightrag.integration.config import FiboLightRAGConfig

class FinancialAnalyzer:
    """Advanced financial document analyzer using FIBO-LightRAG."""
    
    def __init__(self):
        # Configure system for financial analysis
        config = FiboLightRAGConfig(
            retrieval_method='hybrid',
            vector_weight=0.6,
            graph_weight=0.4,
            max_results=20,
            entity_confidence_threshold=0.3,
            enable_inference=True
        )
        
        self.system = FiboLightRAGSystem(config)
        self.system.initialize()
        
        self.analysis_results = {}
    
    def load_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Load financial documents into the system."""
        print(f"📄 Loading {len(documents)} financial documents...")
        
        for doc in documents:
            success = self.system.add_document(
                content=doc['content'],
                doc_id=doc['id'],
                metadata=doc.get('metadata', {})
            )
            
            status = "✅" if success else "❌"
            print(f"   {status} {doc['id']}")
    
    def analyze_company_performance(self, company_name: str) -> Dict[str, Any]:
        """Analyze a specific company's financial performance."""
        print(f"\n🏢 Analyzing {company_name} performance...")
        
        queries = [
            f"What is {company_name}'s revenue?",
            f"What is {company_name}'s profit margin?",
            f"How did {company_name} perform financially?",
            f"What are {company_name}'s key financial metrics?",
            f"What growth did {company_name} achieve?"
        ]
        
        analysis = {
            'company': company_name,
            'metrics': {},
            'performance_summary': [],
            'key_insights': []
        }
        
        for query in queries:
            response = self.system.query(query)
            
            if response.results:
                top_result = response.results[0]
                analysis['performance_summary'].append({
                    'query': query,
                    'score': top_result.score,
                    'content': top_result.content[:200] + "..." if len(top_result.content) > 200 else top_result.content,
                    'entities': top_result.entities,
                    'relationships': top_result.relationships
                })
        
        return analysis
    
    def compare_companies(self, companies: List[str]) -> Dict[str, Any]:
        """Compare financial performance between companies."""
        print(f"\n📊 Comparing companies: {', '.join(companies)}")
        
        comparison = {
            'companies': companies,
            'comparative_analysis': [],
            'relative_performance': {}
        }
        
        # Generate comparison queries
        for i, company1 in enumerate(companies):
            for company2 in companies[i+1:]:
                query = f"Compare {company1} to {company2} financial performance"
                response = self.system.query(query)
                
                if response.results:
                    comparison['comparative_analysis'].append({
                        'companies': [company1, company2],
                        'analysis': response.results[0].content[:300] + "..." if len(response.results[0].content) > 300 else response.results[0].content,
                        'score': response.results[0].score
                    })
        
        return comparison
    
    def analyze_sector_trends(self, sector: str) -> Dict[str, Any]:
        """Analyze trends in a specific financial sector."""
        print(f"\n📈 Analyzing {sector} sector trends...")
        
        trend_queries = [
            f"What are the trends in {sector} sector?",
            f"How is {sector} industry performing?",
            f"What are {sector} sector challenges?",
            f"What growth opportunities exist in {sector}?",
            f"What risks face the {sector} sector?"
        ]
        
        trends = {
            'sector': sector,
            'trend_analysis': [],
            'key_trends': [],
            'opportunities': [],
            'risks': []
        }
        
        for query in trend_queries:
            response = self.system.query(query)
            
            if response.results:
                for result in response.results[:3]:  # Top 3 results
                    trends['trend_analysis'].append({
                        'query': query,
                        'content': result.content,
                        'score': result.score,
                        'source': result.source_type
                    })
        
        return trends
    
    def extract_financial_relationships(self) -> Dict[str, Any]:
        """Extract and analyze financial relationships from the knowledge graph."""
        print("\n🔗 Analyzing financial relationships...")
        
        stats = self.system.get_statistics()
        kg_stats = stats.get('knowledge_graph', {})
        
        relationships = {
            'total_entities': kg_stats.get('num_nodes', 0),
            'total_relationships': kg_stats.get('num_edges', 0),
            'entity_types': kg_stats.get('node_types', {}),
            'relationship_types': kg_stats.get('relationship_types', []),
            'key_relationships': []
        }
        
        # Query for specific relationship types
        relationship_queries = [
            "What companies own other companies?",
            "Which companies have partnerships?",
            "What banks provide services to companies?",
            "Which organizations are competitors?",
            "What investment relationships exist?"
        ]
        
        for query in relationship_queries:
            response = self.system.query(query, method='graph')  # Use graph search for relationships
            
            if response.results:
                relationships['key_relationships'].append({
                    'type': query,
                    'findings': [r.content[:150] + "..." if len(r.content) > 150 else r.content 
                               for r in response.results[:3]]
                })
        
        return relationships
    
    def generate_investment_insights(self) -> Dict[str, Any]:
        """Generate investment insights based on the analyzed data."""
        print("\n💡 Generating investment insights...")
        
        insight_queries = [
            "Which companies show strong revenue growth?",
            "What companies have improving profit margins?",
            "Which stocks have positive performance indicators?",
            "What are the best investment opportunities?",
            "Which companies have strong financial ratios?"
        ]
        
        insights = {
            'investment_opportunities': [],
            'growth_companies': [],
            'financial_strength_indicators': [],
            'market_outlook': []
        }
        
        for query in insight_queries:
            response = self.system.query(query)
            
            if response.results:
                for result in response.results[:2]:  # Top 2 results per query
                    insights['investment_opportunities'].append({
                        'insight': query,
                        'analysis': result.content,
                        'confidence': result.score,
                        'entities_mentioned': result.entities
                    })
        
        return insights
    
    def run_comprehensive_analysis(self, output_file: str = None) -> Dict[str, Any]:
        """Run comprehensive financial analysis and generate report."""
        print("\n🔍 Running comprehensive financial analysis...")
        
        # Get system statistics
        stats = self.system.get_statistics()
        
        # Identify companies mentioned in documents
        company_response = self.system.query("What companies are mentioned in the documents?")
        companies = []
        if company_response.results:
            # Extract companies from entities (simplified)
            for result in company_response.results:
                companies.extend(result.entities)
        
        # Remove duplicates and limit to top companies
        companies = list(set(companies))[:5]
        
        # Run analyses
        comprehensive_report = {
            'executive_summary': {
                'total_documents': stats.get('vector_store', {}).get('total_documents', 0),
                'entities_discovered': stats.get('knowledge_graph', {}).get('num_nodes', 0),
                'relationships_found': stats.get('knowledge_graph', {}).get('num_edges', 0),
                'companies_analyzed': len(companies)
            },
            'company_analyses': [],
            'sector_trends': {},
            'financial_relationships': {},
            'investment_insights': {},
            'recommendations': []
        }
        
        # Analyze individual companies
        for company in companies[:3]:  # Analyze top 3 companies
            if company and len(company) > 2:  # Basic validation
                company_analysis = self.analyze_company_performance(company)
                comprehensive_report['company_analyses'].append(company_analysis)
        
        # Analyze sectors (if identifiable)
        if 'banking' in str(stats).lower() or any('bank' in c.lower() for c in companies):
            comprehensive_report['sector_trends']['banking'] = self.analyze_sector_trends('banking')
        
        if 'tech' in str(stats).lower() or any('tech' in c.lower() for c in companies):
            comprehensive_report['sector_trends']['technology'] = self.analyze_sector_trends('technology')
        
        # Extract relationships
        comprehensive_report['financial_relationships'] = self.extract_financial_relationships()
        
        # Generate investment insights
        comprehensive_report['investment_insights'] = self.generate_investment_insights()
        
        # Generate recommendations
        comprehensive_report['recommendations'] = self._generate_recommendations(comprehensive_report)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            print(f"📄 Report saved to {output_file}")
        
        return comprehensive_report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Based on number of entities and relationships
        if report['executive_summary']['entities_discovered'] > 20:
            recommendations.append("Rich entity network identified - consider deeper relationship analysis")
        
        if report['executive_summary']['relationships_found'] > 50:
            recommendations.append("Complex relationship patterns found - investigate partnership opportunities")
        
        # Based on company analyses
        strong_performers = []
        for analysis in report['company_analyses']:
            if any(result['score'] > 0.8 for result in analysis.get('performance_summary', [])):
                strong_performers.append(analysis['company'])
        
        if strong_performers:
            recommendations.append(f"Strong performance indicators for: {', '.join(strong_performers)}")
        
        # Based on investment insights
        if report['investment_insights'].get('investment_opportunities'):
            recommendations.append("Multiple investment opportunities identified - review detailed analysis")
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Consider adding more financial documents for deeper analysis",
                "Explore specific sector trends for targeted insights",
                "Review entity relationships for partnership opportunities"
            ])
        
        return recommendations

def create_sample_data():
    """Create sample financial documents for analysis."""
    return [
        {
            'id': 'goldman_sachs_2023',
            'content': """
            Goldman Sachs Group Inc. Fourth Quarter and Full Year 2023 Earnings Report
            
            Goldman Sachs reported net revenues of $10.87 billion for the fourth quarter of 2023,
            compared to $10.59 billion in Q4 2022. Full year net revenues were $46.25 billion.
            
            Investment Banking revenues were $2.05 billion in Q4, up 35% from the prior year.
            Global Markets generated $3.92 billion in revenues for the quarter.
            
            The firm's return on equity (ROE) was 8.3% for the full year 2023.
            Book value per share increased to $295.41 at year-end.
            
            Goldman Sachs maintains strategic partnerships with major technology companies
            and continues to expand its digital banking platform, Marcus.
            """,
            'metadata': {
                'company': 'Goldman Sachs',
                'document_type': 'earnings_report',
                'year': 2023,
                'quarter': 'Q4'
            }
        },
        {
            'id': 'jpmorgan_annual_2023',
            'content': """
            JPMorgan Chase & Co. 2023 Annual Report
            
            JPMorgan Chase reported record full-year 2023 net income of $49.6 billion,
            or $15.92 per share. Net revenue was $162.4 billion for the year.
            
            The Consumer & Community Banking segment earned $5.3 billion in net income.
            Corporate & Investment Bank delivered net income of $15.2 billion.
            
            JPMorgan's CET1 ratio was 15.0% at year-end, well above regulatory requirements.
            The bank returned $29.2 billion to shareholders through dividends and buybacks.
            
            The bank completed the acquisition of First Republic Bank and continues
            to invest heavily in technology and digital capabilities.
            """,
            'metadata': {
                'company': 'JPMorgan Chase',
                'document_type': 'annual_report',
                'year': 2023
            }
        },
        {
            'id': 'microsoft_q1_2024',
            'content': """
            Microsoft Corporation Q1 FY2024 Earnings Release
            
            Microsoft delivered record Q1 revenue of $56.5 billion, up 13% year-over-year.
            Operating income increased 25% to $26.9 billion.
            
            Productivity and Business Processes revenue grew 13% to $18.6 billion.
            More Personal Computing revenue was $13.7 billion, up 3%.
            Intelligent Cloud revenue increased 19% to $24.3 billion.
            
            Microsoft Cloud revenue was $31.8 billion, up 24% year-over-year.
            Azure and other cloud services revenue grew 29%.
            
            The company continues to lead in AI innovation with Copilot integrations
            across its product portfolio and maintains partnerships with OpenAI.
            """,
            'metadata': {
                'company': 'Microsoft',
                'document_type': 'earnings_release',
                'year': 2024,
                'quarter': 'Q1'
            }
        },
        {
            'id': 'banking_sector_outlook_2024',
            'content': """
            Banking Sector Outlook 2024 - Industry Analysis
            
            The banking sector faces a mixed environment in 2024. Interest rate stabilization
            should benefit net interest margins, but credit quality concerns persist.
            
            Large banks like JPMorgan Chase and Goldman Sachs are well-positioned due to
            their diversified revenue streams and strong capital positions.
            
            Technology investments remain critical, with banks increasing spending on
            AI and digital transformation. Partnerships with fintech companies are
            becoming increasingly important for competitive advantage.
            
            Regulatory scrutiny continues to intensify, particularly around capital
            requirements and stress testing. ESG considerations are also gaining prominence.
            
            Key trends include the rise of digital banking, increased cybersecurity
            investments, and growing importance of data analytics capabilities.
            """,
            'metadata': {
                'document_type': 'sector_analysis',
                'sector': 'banking',
                'year': 2024
            }
        }
    ]

def main():
    """Run the financial analyzer example."""
    print("🏦 FIBO-LightRAG Financial Analyzer")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = FinancialAnalyzer()
        
        # Load sample data
        documents = create_sample_data()
        analyzer.load_documents(documents)
        
        # Run comprehensive analysis
        report = analyzer.run_comprehensive_analysis('financial_analysis_report.json')
        
        # Display summary
        print("\n📊 Analysis Summary")
        print("-" * 30)
        
        summary = report['executive_summary']
        print(f"Documents Analyzed: {summary['total_documents']}")
        print(f"Financial Entities: {summary['entities_discovered']}")
        print(f"Relationships Found: {summary['relationships_found']}")
        print(f"Companies Analyzed: {summary['companies_analyzed']}")
        
        print("\n💼 Company Performance Highlights:")
        for analysis in report['company_analyses']:
            company = analysis['company']
            highlights = len(analysis.get('performance_summary', []))
            print(f"  • {company}: {highlights} performance metrics analyzed")
        
        print("\n🔗 Financial Relationships:")
        relationships = report['financial_relationships']
        print(f"  • Total Entities: {relationships['total_entities']}")
        print(f"  • Total Relationships: {relationships['total_relationships']}")
        print(f"  • Relationship Types: {len(relationships['relationship_types'])}")
        
        print("\n🎯 Key Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n✅ Complete analysis saved to 'financial_analysis_report.json'")
        print("\n🎉 Financial analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()