"""
Financial document processing for chunking and metric extraction.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of a financial document."""
    id: str
    content: str
    document_id: str
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]

@dataclass
class FinancialMetric:
    """Represents an extracted financial metric."""
    name: str
    value: str
    unit: str
    context: str
    confidence: float

class FinancialDocumentProcessor:
    """Processes financial documents for analysis and retrieval."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.financial_patterns = self._build_financial_patterns()
        
    def _build_financial_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for financial metrics extraction."""
        patterns = {
            'monetary_values': [
                r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?',
                r'(?:USD|EUR|GBP)\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
                r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*dollars?\b'
            ],
            'percentages': [
                r'\b\d+(?:\.\d+)?%',
                r'\b\d+(?:\.\d+)?\s*percent\b'
            ],
            'financial_ratios': [
                r'P/E\s*(?:ratio)?\s*:?\s*\d+(?:\.\d+)?',
                r'ROE\s*:?\s*\d+(?:\.\d+)?%?',
                r'ROI\s*:?\s*\d+(?:\.\d+)?%?',
                r'debt-to-equity\s*:?\s*\d+(?:\.\d+)?'
            ],
            'quarters': [
                r'Q[1-4]\s+\d{4}',
                r'(?:first|second|third|fourth)\s+quarter\s+\d{4}'
            ],
            'years': [
                r'FY\s*\d{4}',
                r'fiscal\s+year\s+\d{4}',
                r'calendar\s+year\s+\d{4}'
            ]
        }
        return patterns
    
    def process_document(self, content: str, document_id: str) -> Tuple[List[DocumentChunk], List[FinancialMetric]]:
        """Process a financial document into chunks and extract metrics."""
        # Clean and normalize content
        cleaned_content = self._clean_content(content)
        
        # Create chunks
        chunks = self._create_chunks(cleaned_content, document_id)
        
        # Extract financial metrics
        metrics = self._extract_financial_metrics(cleaned_content)
        
        logger.info(f"Processed document {document_id}: {len(chunks)} chunks, {len(metrics)} metrics")
        return chunks, metrics
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere with processing
        content = re.sub(r'[^\w\s\.,;:!?()$%/-]', '', content)
        
        # Normalize currency symbols
        content = re.sub(r'[$]', '$', content)
        
        return content.strip()
    
    def _create_chunks(self, content: str, document_id: str) -> List[DocumentChunk]:
        """Create overlapping chunks from document content."""
        chunks = []
        content_length = len(content)
        
        chunk_index = 0
        start_pos = 0
        
        while start_pos < content_length:
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, content_length)
            
            # Try to break at sentence boundary if not at end of document
            if end_pos < content_length:
                # Look for sentence ending within last 100 characters
                sentence_end = content.rfind('.', start_pos, end_pos)
                if sentence_end > start_pos + self.chunk_size // 2:
                    end_pos = sentence_end + 1
            
            # Extract chunk content
            chunk_content = content[start_pos:end_pos].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk_id = f"{document_id}_chunk_{chunk_index}"
                
                # Extract metadata for this chunk
                metadata = self._extract_chunk_metadata(chunk_content)
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next position with overlap
            start_pos = end_pos - self.chunk_overlap
            
            # Ensure we make progress
            if start_pos >= end_pos:
                start_pos = end_pos
        
        return chunks
    
    def _extract_chunk_metadata(self, chunk_content: str) -> Dict[str, Any]:
        """Extract metadata from a chunk."""
        metadata = {
            'length': len(chunk_content),
            'word_count': len(chunk_content.split()),
            'has_monetary_values': False,
            'has_percentages': False,
            'has_dates': False,
            'financial_density': 0.0
        }
        
        # Check for financial content
        chunk_lower = chunk_content.lower()
        
        financial_keywords = [
            'revenue', 'profit', 'loss', 'earnings', 'income', 'expense',
            'asset', 'liability', 'equity', 'debt', 'investment', 'return',
            'cash', 'flow', 'margin', 'ratio', 'growth', 'performance'
        ]
        
        financial_count = sum(1 for keyword in financial_keywords if keyword in chunk_lower)
        metadata['financial_density'] = financial_count / len(financial_keywords)
        
        # Check for specific patterns
        for pattern_type, patterns in self.financial_patterns.items():
            for pattern in patterns:
                if re.search(pattern, chunk_content, re.IGNORECASE):
                    if pattern_type == 'monetary_values':
                        metadata['has_monetary_values'] = True
                    elif pattern_type == 'percentages':
                        metadata['has_percentages'] = True
                    elif pattern_type in ['quarters', 'years']:
                        metadata['has_dates'] = True
        
        return metadata
    
    def _extract_financial_metrics(self, content: str) -> List[FinancialMetric]:
        """Extract financial metrics from document content."""
        metrics = []
        
        # Extract monetary values
        metrics.extend(self._extract_monetary_metrics(content))
        
        # Extract ratios and percentages
        metrics.extend(self._extract_ratio_metrics(content))
        
        # Extract growth metrics
        metrics.extend(self._extract_growth_metrics(content))
        
        return metrics
    
    def _extract_monetary_metrics(self, content: str) -> List[FinancialMetric]:
        """Extract monetary value metrics."""
        metrics = []
        
        # Common financial terms that precede monetary values
        metric_terms = [
            'revenue', 'net income', 'profit', 'loss', 'earnings', 'ebitda',
            'assets', 'liabilities', 'equity', 'debt', 'cash', 'investment'
        ]
        
        for term in metric_terms:
            # Pattern: "revenue of $X" or "revenue: $X" or "revenue $X"
            pattern = rf'{term}\s*(?:of|:)?\s*(\$[\d,]+(?:\.\d{{2}})?(?:\s*(?:million|billion|trillion|M|B|T))?)'
            
            for match in re.finditer(pattern, content, re.IGNORECASE):
                value = match.group(1)
                
                # Get context
                start_context = max(0, match.start() - 50)
                end_context = min(len(content), match.end() + 50)
                context = content[start_context:end_context]
                
                metric = FinancialMetric(
                    name=term.title(),
                    value=value,
                    unit='USD',
                    context=context,
                    confidence=0.8
                )
                metrics.append(metric)
        
        return metrics
    
    def _extract_ratio_metrics(self, content: str) -> List[FinancialMetric]:
        """Extract financial ratio metrics."""
        metrics = []
        
        ratio_patterns = {
            'P/E Ratio': r'P/E\s*(?:ratio)?\s*:?\s*(\d+(?:\.\d+)?)',
            'ROE': r'ROE\s*:?\s*(\d+(?:\.\d+)?%?)',
            'ROI': r'ROI\s*:?\s*(\d+(?:\.\d+)?%?)',
            'Debt-to-Equity': r'debt-to-equity\s*:?\s*(\d+(?:\.\d+)?)',
            'Current Ratio': r'current\s+ratio\s*:?\s*(\d+(?:\.\d+)?)',
            'Quick Ratio': r'quick\s+ratio\s*:?\s*(\d+(?:\.\d+)?)'
        }
        
        for metric_name, pattern in ratio_patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                value = match.group(1)
                
                # Get context
                start_context = max(0, match.start() - 50)
                end_context = min(len(content), match.end() + 50)
                context = content[start_context:end_context]
                
                # Determine unit
                unit = '%' if '%' in value else 'ratio'
                
                metric = FinancialMetric(
                    name=metric_name,
                    value=value,
                    unit=unit,
                    context=context,
                    confidence=0.9
                )
                metrics.append(metric)
        
        return metrics
    
    def _extract_growth_metrics(self, content: str) -> List[FinancialMetric]:
        """Extract growth and change metrics."""
        metrics = []
        
        # Pattern for growth mentions: "revenue growth of 15%" or "increased by 20%"
        growth_pattern = r'(?:growth|increase|decrease|change)\s+(?:of|by)\s+(\d+(?:\.\d+)?%)'
        
        for match in re.finditer(growth_pattern, content, re.IGNORECASE):
            value = match.group(1)
            
            # Get context to determine what grew
            start_context = max(0, match.start() - 100)
            end_context = min(len(content), match.end() + 50)
            context = content[start_context:end_context]
            
            # Try to identify what metric this growth refers to
            context_lower = context.lower()
            metric_name = "Growth"
            
            if 'revenue' in context_lower:
                metric_name = "Revenue Growth"
            elif 'profit' in context_lower:
                metric_name = "Profit Growth"
            elif 'earnings' in context_lower:
                metric_name = "Earnings Growth"
            
            metric = FinancialMetric(
                name=metric_name,
                value=value,
                unit='%',
                context=context,
                confidence=0.7
            )
            metrics.append(metric)
        
        return metrics
    
    def get_document_summary(self, chunks: List[DocumentChunk], metrics: List[FinancialMetric]) -> Dict[str, Any]:
        """Generate a summary of processed document."""
        total_length = sum(chunk.metadata['length'] for chunk in chunks)
        total_words = sum(chunk.metadata['word_count'] for chunk in chunks)
        
        financial_chunks = sum(1 for chunk in chunks if chunk.metadata['financial_density'] > 0.1)
        
        return {
            'total_chunks': len(chunks),
            'total_length': total_length,
            'total_words': total_words,
            'financial_chunks': financial_chunks,
            'extracted_metrics': len(metrics),
            'metrics_by_type': self._categorize_metrics(metrics),
            'average_chunk_size': total_length / len(chunks) if chunks else 0,
            'financial_density': financial_chunks / len(chunks) if chunks else 0
        }
    
    def _categorize_metrics(self, metrics: List[FinancialMetric]) -> Dict[str, int]:
        """Categorize extracted metrics by type."""
        categories = {
            'monetary': 0,
            'ratios': 0,
            'percentages': 0,
            'growth': 0,
            'other': 0
        }
        
        for metric in metrics:
            if metric.unit == 'USD':
                categories['monetary'] += 1
            elif metric.unit == 'ratio':
                categories['ratios'] += 1
            elif metric.unit == '%':
                if 'growth' in metric.name.lower():
                    categories['growth'] += 1
                else:
                    categories['percentages'] += 1
            else:
                categories['other'] += 1
        
        return categories