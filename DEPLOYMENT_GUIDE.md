# FIBO-LightRAG Deployment Guide

## 🚀 Quick Start

### 1. Installation

```bash
# Install Python dependencies
pip install openai anthropic google-generativeai sentence-transformers numpy

# Download the system files
# You need: fibo_lightrag_llm.py and test_llm_system.py
```

### 2. API Key Setup

Choose your preferred LLM provider and set the corresponding API key:

```bash
# OpenAI (recommended for production)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"
```

### 3. Basic Usage

```python
from fibo_lightrag_llm import FiboLightRAGLLMSystem, OpenAIProvider

# Initialize with your preferred provider
llm_provider = OpenAIProvider()  # or AnthropicProvider(), GoogleProvider()
system = FiboLightRAGLLMSystem(llm_provider=llm_provider)

# Add financial documents
system.add_document("quarterly_report", "Your financial document content here...")

# Query for insights
results = system.query("What was the revenue growth this quarter?")
print(results['llm_analysis'])
```

## 🏗️ Architecture Overview

### Core Components

1. **LLM Providers** (`OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`)
   - Handle text generation and embeddings
   - Configurable models and parameters

2. **FIBO Parser** (`LLMEnhancedFiboParser`)
   - Loads financial ontology classes
   - LLM-enhanced descriptions and examples

3. **Entity Extractor** (`LLMEntityExtractor`)
   - Uses LLM to identify financial entities
   - Maps entities to FIBO ontology classes

4. **Knowledge Graph Builder** (`LLMGraphBuilder`)
   - Creates nodes for extracted entities
   - Uses LLM to infer relationships

5. **Vector Store** (`LLMVectorStore`)
   - Stores document embeddings
   - Enables semantic similarity search

6. **Retrieval Engine** (`LLMRetrievalEngine`)
   - Dual-level retrieval (vector + graph)
   - Context-aware result ranking

## 🔧 Configuration Options

### LLM Provider Configuration

```python
# OpenAI Configuration
provider = OpenAIProvider(
    model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo"
    embedding_model="text-embedding-3-large"
)

# Anthropic Configuration  
provider = AnthropicProvider(
    model="claude-3-opus-20240229"  # or "claude-3-sonnet-20240229"
)

# Custom Local Provider
class LocalProvider(LLMProvider):
    def generate_text(self, prompt, max_tokens=1000, temperature=0.1):
        # Your local model implementation
        pass
```

### System Configuration

```python
system = FiboLightRAGLLMSystem(
    llm_provider=provider,
    chunk_size=1000,          # Document chunk size
    chunk_overlap=200,        # Overlap between chunks
    max_entities_per_chunk=50, # Max entities to extract per chunk
    confidence_threshold=0.1,  # Minimum confidence for results
    embedding_dimension=1536   # Embedding vector dimension
)
```

## 📊 Production Deployment

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv fibo-lightrag-env
source fibo-lightrag-env/bin/activate  # Linux/Mac
# fibo-lightrag-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fibo_lightrag_llm.py .
COPY your_app.py .

CMD ["python", "your_app.py"]
```

### 3. API Service Example

```python
from flask import Flask, request, jsonify
from fibo_lightrag_llm import FiboLightRAGLLMSystem, OpenAIProvider

app = Flask(__name__)

# Initialize system once
llm_provider = OpenAIProvider()
system = FiboLightRAGLLMSystem(llm_provider=llm_provider)

@app.route('/add_document', methods=['POST'])
def add_document():
    data = request.json
    doc_id = data['doc_id']
    content = data['content']
    
    system.add_document(doc_id, content)
    return jsonify({"status": "success", "doc_id": doc_id})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data['query']
    
    results = system.query(query_text)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🔍 Testing & Validation

### 1. Unit Tests

```bash
# Test with mock provider (no API keys needed)
python test_llm_system.py

# Test with real provider
python demo_production.py
```

### 2. Performance Testing

```python
import time

# Test document processing speed
start_time = time.time()
system.add_document("test_doc", large_document_content)
processing_time = time.time() - start_time
print(f"Document processing time: {processing_time:.2f}s")

# Test query performance
start_time = time.time()
results = system.query("complex financial query")
query_time = time.time() - start_time
print(f"Query time: {query_time:.2f}s")
```

## 📈 Monitoring & Optimization

### 1. Performance Metrics

- Document processing time
- Query response time
- LLM API usage and costs
- Graph size and complexity
- Retrieval accuracy

### 2. Cost Optimization

```python
# Use smaller models for development
dev_provider = OpenAIProvider(model="gpt-3.5-turbo")

# Cache LLM responses
system = FiboLightRAGLLMSystem(
    llm_provider=provider,
    enable_caching=True,
    cache_ttl=3600  # 1 hour
)
```

### 3. Error Handling

```python
try:
    results = system.query("financial query")
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Fallback to simpler search
    results = system.vector_search_only("financial query")
```

## 🔒 Security Considerations

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Monitor usage

2. **Data Privacy**
   - Encrypt sensitive documents
   - Implement access controls
   - Log audit trails

3. **Rate Limiting**
   - Implement request throttling
   - Monitor API quotas
   - Handle rate limit errors

## 🚨 Troubleshooting

### Common Issues

1. **"No entities extracted"**
   - Check document content quality
   - Verify LLM provider is working
   - Lower confidence threshold

2. **"No relationships found"**
   - Ensure entities are close in text
   - Check relationship inference prompts
   - Increase context window

3. **"API errors"**
   - Verify API keys
   - Check rate limits
   - Monitor error logs

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
system = FiboLightRAGLLMSystem(
    llm_provider=provider,
    debug_mode=True
)
```

## 📞 Support

For technical support and questions:
1. Check the logs for detailed error messages
2. Review the API documentation
3. Test with mock provider first
4. Monitor LLM provider status

## 🔄 Updates & Maintenance

1. **Regular Updates**
   - Update LLM provider models
   - Refresh FIBO ontology
   - Monitor performance metrics

2. **Backup Strategy**
   - Export knowledge graphs
   - Backup document indices
   - Version control configurations