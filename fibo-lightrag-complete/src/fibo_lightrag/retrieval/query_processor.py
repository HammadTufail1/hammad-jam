"""
Query processing and understanding for financial queries.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of financial queries."""
    FACTUAL = "factual"  # What is X?
    COMPARATIVE = "comparative"  # How does X compare to Y?
    TEMPORAL = "temporal"  # How has X changed over time?
    RELATIONSHIP = "relationship"  # What is the relationship between X and Y?
    ANALYTICAL = "analytical"  # Why did X happen?
    PREDICTIVE = "predictive"  # What will happen to X?

@dataclass
class QueryIntent:
    """Parsed query intent and components."""
    query_type: QueryType
    entities: List[str]
    temporal_indicators: List[str]
    financial_metrics: List[str]
    comparison_targets: List[str]
    relationship_types: List[str]
    confidence: float

class FinancialQueryProcessor:
    """Processes and understands financial queries."""
    
    def __init__(self):
        self.query_patterns = self._build_query_patterns()
        self.financial_terms = self._build_financial_terms()
        self.temporal_terms = self._build_temporal_terms()
        self.comparison_terms = self._build_comparison_terms()
        self.relationship_terms = self._build_relationship_terms()
    
    def _build_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Build regex patterns for different query types."""
        patterns = {
            QueryType.FACTUAL: [
                r'\bwhat\s+is\b',
                r'\bwho\s+is\b',
                r'\bwhere\s+is\b',
                r'\bdefine\b',
                r'\bexplain\b',
                r'\btell\s+me\s+about\b'
            ],
            QueryType.COMPARATIVE: [
                r'\bcompare\b',
                r'\bvs\b',
                r'\bversus\b',
                r'\bdifference\s+between\b',
                r'\bbetter\s+than\b',
                r'\bhigher\s+than\b',
                r'\blower\s+than\b'
            ],
            QueryType.TEMPORAL: [
                r'\bover\s+time\b',
                r'\bhistorical\b',
                r'\btrend\b',
                r'\bchanged?\b',
                r'\bgrowth\b',
                r'\bdecline\b',
                r'\byear\s+over\s+year\b',
                r'\bquarter\s+over\s+quarter\b'
            ],
            QueryType.RELATIONSHIP: [
                r'\brelationship\s+between\b',
                r'\bconnection\s+between\b',
                r'\bhow\s+does.*relate\b',
                r'\bimpact\s+of\b',
                r'\beffect\s+of\b',
                r'\binfluence\s+of\b'
            ],
            QueryType.ANALYTICAL: [
                r'\bwhy\b',
                r'\breason\b',
                r'\bcause\b',
                r'\bdriver\b',
                r'\bfactor\b',
                r'\bexplain\s+why\b'
            ],
            QueryType.PREDICTIVE: [
                r'\bwill\b',
                r'\bpredict\b',
                r'\bforecast\b',
                r'\bexpected?\b',
                r'\bfuture\b',
                r'\boutlook\b',
                r'\bprojection\b'
            ]
        }
        return patterns
    
    def _build_financial_terms(self) -> Set[str]:
        """Build set of financial terms for entity recognition."""
        return {
            # Financial metrics
            'revenue', 'profit', 'loss', 'earnings', 'income', 'ebitda',
            'margin', 'ratio', 'return', 'yield', 'dividend', 'eps',
            'pe ratio', 'roe', 'roi', 'roa', 'debt', 'equity', 'assets',
            'liabilities', 'cash flow', 'free cash flow', 'capex',
            
            # Financial instruments
            'stock', 'bond', 'security', 'share', 'option', 'future',
            'derivative', 'loan', 'mortgage', 'credit', 'deposit',
            'investment', 'fund', 'etf', 'mutual fund', 'hedge fund',
            
            # Market terms
            'market cap', 'volume', 'price', 'volatility', 'beta',
            'correlation', 'valuation', 'multiple', 'premium', 'discount',
            
            # Business terms
            'acquisition', 'merger', 'partnership', 'subsidiary',
            'joint venture', 'ipo', 'spinoff', 'restructuring'
        }
    
    def _build_temporal_terms(self) -> Set[str]:
        """Build set of temporal indicators."""
        return {
            'q1', 'q2', 'q3', 'q4', 'quarter', 'quarterly',
            'annual', 'yearly', 'monthly', 'weekly', 'daily',
            'fy', 'fiscal year', 'calendar year', 'ytd', 'year to date',
            '2023', '2024', '2025', 'last year', 'this year', 'next year',
            'historical', 'trend', 'over time', 'period', 'duration'
        }
    
    def _build_comparison_terms(self) -> Set[str]:
        """Build set of comparison indicators."""
        return {
            'vs', 'versus', 'compared to', 'relative to', 'against',
            'higher', 'lower', 'better', 'worse', 'more', 'less',
            'increase', 'decrease', 'growth', 'decline', 'change',
            'difference', 'ratio', 'multiple', 'percentage'
        }
    
    def _build_relationship_terms(self) -> Set[str]:
        """Build set of relationship indicators."""
        return {
            'owns', 'owned by', 'subsidiary', 'parent', 'partner',
            'partnership', 'alliance', 'joint venture', 'collaboration',
            'acquisition', 'merger', 'spinoff', 'divested',
            'provides', 'offers', 'serves', 'customers', 'clients',
            'supplies', 'vendor', 'supplier', 'distributor'
        }
    
    def process_query(self, query: str) -> QueryIntent:
        """Process a query and extract intent and components."""
        query_lower = query.lower()
        
        # Determine query type
        query_type, type_confidence = self._classify_query_type(query_lower)
        
        # Extract entities (simplified - in practice, would use NER)
        entities = self._extract_query_entities(query)
        
        # Extract temporal indicators
        temporal_indicators = self._extract_temporal_indicators(query_lower)
        
        # Extract financial metrics
        financial_metrics = self._extract_financial_metrics(query_lower)
        
        # Extract comparison targets
        comparison_targets = self._extract_comparison_targets(query_lower)
        
        # Extract relationship types
        relationship_types = self._extract_relationship_types(query_lower)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            type_confidence, entities, temporal_indicators, 
            financial_metrics, comparison_targets, relationship_types
        )
        
        intent = QueryIntent(
            query_type=query_type,
            entities=entities,
            temporal_indicators=temporal_indicators,
            financial_metrics=financial_metrics,
            comparison_targets=comparison_targets,
            relationship_types=relationship_types,
            confidence=confidence
        )
        
        logger.debug(f"Processed query: {query} -> {intent}")
        return intent
    
    def _classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Classify the type of query."""
        type_scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            
            if score > 0:
                type_scores[query_type] = score / len(patterns)
        
        if not type_scores:
            return QueryType.FACTUAL, 0.5  # Default
        
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]
        
        return best_type, confidence
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        entities = []
        
        # Simple extraction - look for capitalized words/phrases
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', query)
        entities.extend(capitalized_words)
        
        # Look for quoted entities
        quoted_entities = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted_entities)
        
        # Look for financial terms in the query
        query_lower = query.lower()
        for term in self.financial_terms:
            if term in query_lower:
                entities.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query."""
        indicators = []
        
        for term in self.temporal_terms:
            if term in query:
                indicators.append(term)
        
        # Extract specific years
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        indicators.extend(years)
        
        return indicators
    
    def _extract_financial_metrics(self, query: str) -> List[str]:
        """Extract financial metrics mentioned in query."""
        metrics = []
        
        for term in self.financial_terms:
            if term in query:
                metrics.append(term)
        
        return metrics
    
    def _extract_comparison_targets(self, query: str) -> List[str]:
        """Extract comparison targets from query."""
        targets = []
        
        # Look for comparison patterns
        comparison_patterns = [
            r'compare\s+([^.]+?)\s+(?:to|with|and)\s+([^.]+)',
            r'([^.]+?)\s+vs\s+([^.]+)',
            r'([^.]+?)\s+versus\s+([^.]+)',
            r'difference\s+between\s+([^.]+?)\s+and\s+([^.]+)'
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                targets.extend([item.strip() for item in match])
        
        return targets
    
    def _extract_relationship_types(self, query: str) -> List[str]:
        """Extract relationship types from query."""
        relationships = []
        
        for term in self.relationship_terms:
            if term in query:
                relationships.append(term)
        
        return relationships
    
    def _calculate_confidence(self, type_confidence: float, entities: List[str], 
                            temporal_indicators: List[str], financial_metrics: List[str],
                            comparison_targets: List[str], relationship_types: List[str]) -> float:
        """Calculate overall confidence in query understanding."""
        base_confidence = type_confidence
        
        # Boost confidence based on extracted components
        if entities:
            base_confidence += 0.1
        if financial_metrics:
            base_confidence += 0.1
        if temporal_indicators:
            base_confidence += 0.05
        if comparison_targets:
            base_confidence += 0.05
        if relationship_types:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def generate_search_terms(self, intent: QueryIntent) -> List[str]:
        """Generate search terms based on query intent."""
        search_terms = []
        
        # Add entities
        search_terms.extend(intent.entities)
        
        # Add financial metrics
        search_terms.extend(intent.financial_metrics)
        
        # Add comparison targets
        search_terms.extend(intent.comparison_targets)
        
        # Add relationship-specific terms
        if intent.query_type == QueryType.RELATIONSHIP:
            search_terms.extend(intent.relationship_types)
        
        # Add temporal context if relevant
        if intent.query_type == QueryType.TEMPORAL:
            search_terms.extend(intent.temporal_indicators)
        
        # Remove duplicates and empty strings
        unique_terms = list(set(term.strip() for term in search_terms if term.strip()))
        
        logger.debug(f"Generated search terms: {unique_terms}")
        return unique_terms
    
    def suggest_refinements(self, intent: QueryIntent) -> List[str]:
        """Suggest query refinements based on intent analysis."""
        suggestions = []
        
        if intent.confidence < 0.6:
            suggestions.append("Consider being more specific about the financial entities or metrics you're interested in.")
        
        if intent.query_type == QueryType.COMPARATIVE and len(intent.comparison_targets) < 2:
            suggestions.append("For comparisons, try specifying exactly what you want to compare.")
        
        if intent.query_type == QueryType.TEMPORAL and not intent.temporal_indicators:
            suggestions.append("Consider specifying a time period (e.g., 'in 2023', 'over the last quarter').")
        
        if not intent.financial_metrics and intent.query_type != QueryType.FACTUAL:
            suggestions.append("Try including specific financial metrics (e.g., revenue, profit, ROE).")
        
        return suggestions