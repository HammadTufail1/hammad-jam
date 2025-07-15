# FIBO-LightRAG: Financial Knowledge Graph System

A comprehensive system that combines LightRAG concepts with Financial Industry Business Ontology (FIBO) for advanced financial document analysis using LLMs and embeddings.

## Overview

FIBO-LightRAG creates a specialized knowledge graph for financial documents by:
- Parsing FIBO ontologies to understand financial entity relationships
- Extracting financial entities from documents using multiple strategies
- Building knowledge graphs with inferred relationships
- Providing dual-level retrieval (vector + graph-based)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download FIBO ontologies (optional - sample data included)
python src/fibo_lightrag/fibo/download_fibo.py

# Run the demo
python demo_system.py

# Check system status
python check_system.py
```

## Architecture

```
fibo-lightrag/
├── src/fibo_lightrag/           # Core system
│   ├── fibo/                    # FIBO ontology parsing
│   ├── lightrag/                # Entity extraction
│   ├── graph/                   # Knowledge graph operations
│   ├── retrieval/               # Document processing & retrieval
│   └── integration/             # System integration
├── examples/                    # Usage examples
├── docs/                        # Documentation
├── data/                        # FIBO ontology data
└── tests/                       # Test scripts
```

## Features

- **FIBO Integration**: Parse and utilize Financial Industry Business Ontology
- **Multi-Strategy Entity Extraction**: Dictionary, regex, and context-aware extraction
- **Knowledge Graph Construction**: Build graphs with financial entity relationships
- **Dual-Level Retrieval**: Combine semantic and structural search
- **Financial Document Processing**: Specialized handling for financial reports
- **Inference Engine**: Derive relationships using FIBO rules

## Usage Example

```python
from src.fibo_lightrag.integration.fibo_lightrag_system import FiboLightRAGSystem

# Initialize system
system = FiboLightRAGSystem()

# Process financial document
result = system.process_document("path/to/financial_report.txt")

# Query the system
response = system.query("What partnerships exist between banks?")
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Usage Guide](docs/USAGE_GUIDE.md)
- [Getting Started](GETTING_STARTED.md)

## Requirements

- Python 3.8+
- rdflib for ontology parsing
- Custom vector operations (no heavy ML dependencies)

## License

MIT License - see LICENSE file for details.