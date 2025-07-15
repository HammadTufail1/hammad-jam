# Getting Started with FIBO-LightRAG

Welcome to FIBO-LightRAG! This guide will help you get up and running quickly with the Financial Industry Business Ontology (FIBO) enhanced LightRAG system.

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python check_system.py
```

This will verify that all components are working correctly.

### 3. Run the Demo

```bash
python demo_system.py
```

This will demonstrate the system with sample financial documents.

## Basic Usage

### Initialize the System

```python
from fibo_lightrag import FiboLightRAGSystem

# Create and initialize system
system = FiboLightRAGSystem()
system.initialize()
```

### Add Financial Documents

```python
# Add a financial document
document = """
Apple Inc. reported Q4 2023 revenue of $119.6 billion, 
up 2% year-over-year. The company's services revenue 
reached $22.3 billion, growing 16% compared to the 
previous year quarter.
"""

system.add_document(document, doc_id="apple_q4_2023")
```

### Query the System

```python
# Query for information
response = system.query("What was Apple's Q4 2023 revenue?")

# Access results
for result in response.results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content}")
    print(f"Source: {result.source_type}")
```

## Understanding Results

### Retrieval Methods

FIBO-LightRAG supports three retrieval methods:

1. **Vector Search** (`method='vector'`): Semantic similarity search
2. **Graph Search** (`method='graph'`): Knowledge graph traversal
3. **Hybrid Search** (`method='hybrid'`): Combines both approaches (default)

### Result Components

Each result includes:
- `content`: The relevant text content
- `score`: Relevance score (0-1)
- `source_type`: How the result was found (vector/graph/hybrid)
- `entities`: Extracted financial entities
- `relationships`: Discovered relationships
- `metadata`: Additional context information

## Configuration

### Custom Configuration

```python
from fibo_lightrag import FiboLightRAGConfig, FiboLightRAGSystem

# Create custom configuration
config = FiboLightRAGConfig(
    retrieval_method='hybrid',
    vector_weight=0.7,
    graph_weight=0.3,
    max_results=15
)

# Use with system
system = FiboLightRAGSystem(config)
```

### Key Configuration Options

- `retrieval_method`: 'vector', 'graph', or 'hybrid'
- `vector_weight` / `graph_weight`: Importance weights for hybrid search
- `max_results`: Maximum number of results to return
- `chunk_size`: Size of document chunks for processing
- `entity_confidence_threshold`: Minimum confidence for entity extraction

## Working with Financial Data

### Supported Document Types

- Annual reports
- Quarterly earnings reports
- SEC filings (10-K, 10-Q, 8-K)
- Market analysis reports
- Financial news articles
- Analyst research reports

### Automatically Extracted Entities

The system automatically extracts:
- **Organizations**: Companies, banks, institutions
- **Financial Instruments**: Stocks, bonds, securities
- **Financial Metrics**: Revenue, profit, ratios
- **Monetary Amounts**: Dollar values, percentages
- **Time Periods**: Quarters, fiscal years

### Relationship Discovery

The system identifies relationships such as:
- Ownership (Company A owns Company B)
- Partnerships (Company A partners with Company B)
- Service provision (Bank A provides services to Company B)
- Competition (Company A competes with Company B)

## Advanced Features

### Query Types

The system understands different types of financial queries:

```python
# Factual queries
system.query("What is JPMorgan Chase's revenue?")

# Comparative queries  
system.query("Compare Apple's revenue to Microsoft's revenue")

# Temporal queries
system.query("How has Tesla's stock price changed over time?")

# Relationship queries
system.query("What is the relationship between Goldman Sachs and Apple?")
```

### Batch Processing

```python
# Process multiple documents
documents = [
    ("Document 1 content...", "doc1", {"type": "earnings"}),
    ("Document 2 content...", "doc2", {"type": "annual_report"}),
]

results = system.add_documents(documents)
```

### System Statistics

```python
# Get system statistics
stats = system.get_statistics()
print(f"Documents: {stats['vector_store']['total_documents']}")
print(f"Graph nodes: {stats['knowledge_graph']['num_nodes']}")
print(f"Graph edges: {stats['knowledge_graph']['num_edges']}")
```

## Best Practices

### Document Preparation

1. **Clean Text**: Remove unnecessary formatting and headers
2. **Consistent Format**: Use consistent date and currency formats
3. **Context**: Include relevant context (company names, time periods)
4. **Metadata**: Add meaningful metadata (document type, year, company)

### Query Formulation

1. **Be Specific**: Use specific company names and metric names
2. **Include Context**: Add time periods when relevant
3. **Use Financial Terms**: Leverage financial vocabulary the system understands
4. **Try Different Phrasings**: Experiment with different ways to ask the same question

### Performance Optimization

1. **Chunk Size**: Adjust chunk size based on document length
2. **Confidence Thresholds**: Tune thresholds based on your data quality
3. **Retrieval Weights**: Adjust vector/graph weights based on your use case
4. **Batch Processing**: Process multiple documents in batches for efficiency

## Troubleshooting

### Common Issues

**No results returned:**
- Check if documents were added successfully
- Lower the confidence threshold
- Try different query phrasings
- Verify entity extraction is working

**Low-quality results:**
- Increase confidence thresholds
- Improve document quality and consistency
- Adjust retrieval method weights
- Add more relevant documents

**Slow performance:**
- Reduce chunk size for large documents
- Lower max_results for faster queries
- Consider using vector-only retrieval for speed

### Getting Help

1. Check the `docs/` folder for detailed documentation
2. Run `python check_system.py` to verify system health
3. Enable debug logging for detailed information
4. Review the demo script for usage examples

## Next Steps

Once you're comfortable with the basics:

1. Explore the API reference in `docs/API_REFERENCE.md`
2. Read the usage guide in `docs/USAGE_GUIDE.md`
3. Try the financial analyzer example in `examples/`
4. Experiment with your own financial documents
5. Customize the configuration for your specific use case

Happy analyzing! 🏦📊