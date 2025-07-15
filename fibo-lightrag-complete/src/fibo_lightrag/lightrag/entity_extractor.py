"""
Entity extraction for financial documents using FIBO ontology.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Represents an extracted financial entity."""
    text: str
    fibo_uri: str
    fibo_class: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str

class FiboEntityExtractor:
    """Extracts financial entities from text using FIBO ontology."""
    
    def __init__(self, fibo_parser):
        self.fibo_parser = fibo_parser
        self.entity_patterns = self._build_entity_patterns()
        self.financial_keywords = self._build_financial_keywords()
        
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for different entity types."""
        patterns = {
            'organizations': [
                r'\b[A-Z][a-zA-Z\s&,.-]+(?:Corp|Corporation|Inc|Incorporated|Ltd|Limited|LLC|Company|Bank|Group|Holdings|Partners)\b',
                r'\b(?:Federal Reserve|SEC|FDIC|OCC|CFTC|FINRA|NASDAQ|NYSE)\b',
                r'\b[A-Z]{2,5}\s+(?:Bank|Credit Union|Insurance|Investment)\b'
            ],
            'financial_instruments': [
                r'\b(?:stock|bond|equity|security|derivative|option|future|swap|note|bill)\b',
                r'\b(?:mortgage|loan|credit|deposit|investment)\b',
                r'\b[A-Z]{3,5}\s+(?:bond|note|security)\b'
            ],
            'monetary_amounts': [
                r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
                r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*dollars?\b'
            ],
            'financial_metrics': [
                r'\b(?:revenue|profit|loss|earnings|EBITDA|ROE|ROI|P/E|debt|equity|assets|liabilities)\b',
                r'\b(?:interest rate|yield|return|dividend|margin|ratio)\b'
            ],
            'dates': [
                r'\b(?:Q[1-4]\s+\d{4}|FY\s*\d{4}|fiscal\s+year\s+\d{4})\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
        }
        return patterns
    
    def _build_financial_keywords(self) -> Dict[str, Set[str]]:
        """Build keyword sets for financial entity classification."""
        keywords = {
            'banking': {'bank', 'banking', 'deposit', 'loan', 'credit', 'savings', 'checking'},
            'investment': {'investment', 'portfolio', 'fund', 'asset', 'equity', 'bond', 'security'},
            'insurance': {'insurance', 'policy', 'premium', 'claim', 'coverage', 'underwriting'},
            'trading': {'trading', 'trade', 'market', 'exchange', 'broker', 'dealer'},
            'regulation': {'regulatory', 'compliance', 'rule', 'regulation', 'filing', 'disclosure'}
        }
        return keywords
    
    def extract_entities(self, text: str, context_window: int = 50) -> List[ExtractedEntity]:
        """Extract financial entities from text."""
        entities = []
        
        # Dictionary-based extraction using FIBO labels
        entities.extend(self._extract_fibo_entities(text, context_window))
        
        # Pattern-based extraction
        entities.extend(self._extract_pattern_entities(text, context_window))
        
        # Context-aware extraction
        entities.extend(self._extract_contextual_entities(text, context_window))
        
        # Remove duplicates and sort by confidence
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _extract_fibo_entities(self, text: str, context_window: int) -> List[ExtractedEntity]:
        """Extract entities using FIBO class labels."""
        entities = []
        text_lower = text.lower()
        
        for uri, fibo_class in self.fibo_parser.classes.items():
            label = fibo_class.label.lower()
            
            # Skip very short labels
            if len(label) < 3:
                continue
            
            # Find all occurrences of the label
            for match in re.finditer(re.escape(label), text_lower):
                start_pos = match.start()
                end_pos = match.end()
                
                # Get context
                context_start = max(0, start_pos - context_window)
                context_end = min(len(text), end_pos + context_window)
                context = text[context_start:context_end]
                
                # Calculate confidence based on context
                confidence = self._calculate_confidence(text, start_pos, end_pos, uri)
                
                entity = ExtractedEntity(
                    text=text[start_pos:end_pos],
                    fibo_uri=uri,
                    fibo_class=fibo_class.label,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context
                )
                entities.append(entity)
        
        return entities
    
    def _extract_pattern_entities(self, text: str, context_window: int) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get context
                    context_start = max(0, start_pos - context_window)
                    context_end = min(len(text), end_pos + context_window)
                    context = text[context_start:context_end]
                    
                    # Map to FIBO class
                    fibo_uri, fibo_class = self._map_to_fibo_class(entity_type, match.group())
                    
                    if fibo_uri:
                        confidence = self._calculate_pattern_confidence(entity_type, match.group())
                        
                        entity = ExtractedEntity(
                            text=match.group(),
                            fibo_uri=fibo_uri,
                            fibo_class=fibo_class,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context
                        )
                        entities.append(entity)
        
        return entities
    
    def _extract_contextual_entities(self, text: str, context_window: int) -> List[ExtractedEntity]:
        """Extract entities using contextual analysis."""
        entities = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            # Look for capitalized phrases that might be entities
            capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z\s&,-]{2,30}\b', sentence)
            
            for phrase in capitalized_phrases:
                # Skip if already extracted
                if any(phrase.lower() in entity.text.lower() for entity in entities):
                    continue
                
                # Check if phrase appears in financial context
                context_keywords = set(re.findall(r'\b\w+\b', sentence.lower()))
                financial_score = 0
                
                for category, keywords in self.financial_keywords.items():
                    if context_keywords & keywords:
                        financial_score += 1
                
                if financial_score > 0:
                    # Try to map to FIBO class
                    fibo_uri, fibo_class = self._map_phrase_to_fibo(phrase)
                    
                    if fibo_uri:
                        # Find position in original text
                        match = re.search(re.escape(phrase), text)
                        if match:
                            start_pos = match.start()
                            end_pos = match.end()
                            
                            context_start = max(0, start_pos - context_window)
                            context_end = min(len(text), end_pos + context_window)
                            context = text[context_start:context_end]
                            
                            confidence = min(0.8, financial_score * 0.2)
                            
                            entity = ExtractedEntity(
                                text=phrase,
                                fibo_uri=fibo_uri,
                                fibo_class=fibo_class,
                                confidence=confidence,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                context=context
                            )
                            entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self, text: str, start_pos: int, end_pos: int, fibo_uri: str) -> float:
        """Calculate confidence score for an extracted entity."""
        base_confidence = 0.7
        
        # Get surrounding context
        context_start = max(0, start_pos - 100)
        context_end = min(len(text), end_pos + 100)
        context = text[context_start:context_end].lower()
        
        # Bonus for financial keywords in context
        financial_keywords = {'financial', 'bank', 'investment', 'market', 'fund', 'asset', 'security'}
        context_bonus = sum(0.05 for keyword in financial_keywords if keyword in context)
        
        # Bonus for proper capitalization
        entity_text = text[start_pos:end_pos]
        if entity_text[0].isupper():
            base_confidence += 0.1
        
        return min(1.0, base_confidence + context_bonus)
    
    def _calculate_pattern_confidence(self, entity_type: str, text: str) -> float:
        """Calculate confidence for pattern-based entities."""
        confidence_map = {
            'organizations': 0.8,
            'financial_instruments': 0.7,
            'monetary_amounts': 0.9,
            'financial_metrics': 0.6,
            'dates': 0.8
        }
        return confidence_map.get(entity_type, 0.5)
    
    def _map_to_fibo_class(self, entity_type: str, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Map entity type and text to FIBO class."""
        # Simple mapping based on entity type
        type_mapping = {
            'organizations': 'Organization',
            'financial_instruments': 'FinancialInstrument',
            'monetary_amounts': 'MonetaryAmount',
            'financial_metrics': 'FinancialMetric',
            'dates': 'Date'
        }
        
        fibo_class_name = type_mapping.get(entity_type)
        if fibo_class_name:
            # Find matching FIBO URI
            for uri, fibo_class in self.fibo_parser.classes.items():
                if fibo_class_name.lower() in fibo_class.label.lower():
                    return uri, fibo_class.label
        
        return None, None
    
    def _map_phrase_to_fibo(self, phrase: str) -> Tuple[Optional[str], Optional[str]]:
        """Map a phrase to the most appropriate FIBO class."""
        phrase_lower = phrase.lower()
        
        # Simple heuristics for mapping
        if any(word in phrase_lower for word in ['bank', 'corp', 'inc', 'ltd', 'company']):
            for uri, fibo_class in self.fibo_parser.classes.items():
                if 'organization' in fibo_class.label.lower():
                    return uri, fibo_class.label
        
        # Default to a generic entity class if available
        for uri, fibo_class in self.fibo_parser.classes.items():
            if 'entity' in fibo_class.label.lower():
                return uri, fibo_class.label
        
        return None, None
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping the highest confidence ones."""
        seen_spans = set()
        unique_entities = []
        
        for entity in entities:
            span = (entity.start_pos, entity.end_pos)
            if span not in seen_spans:
                seen_spans.add(span)
                unique_entities.append(entity)
            else:
                # Keep the entity with higher confidence
                for i, existing in enumerate(unique_entities):
                    if (existing.start_pos, existing.end_pos) == span:
                        if entity.confidence > existing.confidence:
                            unique_entities[i] = entity
                        break
        
        return unique_entities