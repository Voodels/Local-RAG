# Modular RAG Pipeline v2.0

A modular, extensible local RAG (Retrieval Augmented Generation) pipeline that runs entirely on your hardware. This system processes PDF documents, creates embeddings, and uses a local LLM to answer questions based on the document content.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with distinct modules for different functionality
- **Local Processing**: Runs entirely on your hardware - no API calls required
- **GPU Acceleration**: Automatic GPU detection and usage when available
- **Flexible Configuration**: Command-line arguments, config files, and environment variables
- **Multiple Interfaces**: Interactive mode, batch processing, and single queries
- **Progress Tracking**: Rich terminal UI with progress bars and status updates
- **Extensible Design**: Easy to add new document types, embedding models, or LLMs

## 📁 Project Structure

```
rag_pipeline/
├── __init__.py                 # Package initialization
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── pipeline.py            # Main pipeline orchestration
├── processors/
│   ├── __init__.py
│   ├── document.py            # PDF processing and text extraction
│   ├── embedding.py           # Text embedding creation
│   ├── llm.py                 # Language model processing
│   └── retrieval.py           # Vector search and retrieval
├── utils/
│   ├── __init__.py
│   ├── device.py              # GPU/CPU device management
│   ├── logging.py             # Logging configuration
│   └── text.py                # Text processing utilities
└── interface.py               # User interface modes
```

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- 8GB+ RAM (16GB+ recommended for larger models)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/your/project
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Quick Start

### Basic Usage

1. **Interactive Mode** (Recommended for exploration):
   ```bash
   python main.py --interactive
   ```

2. **Single Query**:
   ```bash
   python main.py --query "What are the main nutrients discussed in the document?"
   ```

3. **Sample Queries** (Test with predefined questions):
   ```bash
   python main.py --sample-queries
   ```

### Advanced Usage

**Custom Configuration:**
```bash
python main.py --config config_example.json --interactive
```

**Specify Different Models:**
```bash
python main.py --model microsoft/DialoGPT-medium --embedding-model all-MiniLM-L6-v2 --interactive
```

**Force CPU Usage:**
```bash
python main.py --device cpu --interactive
```

**Custom PDF:**
```bash
python main.py --pdf /path/to/your/document.pdf --interactive
```

## ⚙️ Configuration

### Command Line Options

```bash
python main.py --help
```

**Configuration Options:**
- `--config`: Path to JSON configuration file
- `--pdf`: Path to PDF document
- `--pdf-url`: URL to download PDF if not present
- `--model`: HuggingFace LLM model ID
- `--embedding-model`: Sentence transformer model name
- `--device`: Device preference (auto/cpu/cuda)
- `--no-quantization`: Disable 4-bit quantization
- `--verbose`: Enable verbose output
- `--log-level`: Set logging level (DEBUG/INFO/WARNING/ERROR)

**Execution Modes:**
- `--interactive`: Interactive Q&A mode
- `--sample-queries`: Run predefined sample queries
- `--query "text"`: Process a single query

### Configuration File

Create a `config.json` file (see `config_example.json`):

```json
{
  "pdf_path": "document.pdf",
  "llm_model_id": "microsoft/DialoGPT-medium",
  "embedding_model_name": "all-mpnet-base-v2",
  "use_quantization": true,
  "temperature": 0.7,
  "max_new_tokens": 512,
  "n_resources_to_return": 5,
  "verbose": true
}
```

### Environment Variables

Set environment variables with `RAG_` prefix:

```bash
export RAG_PDF_PATH="document.pdf"
export RAG_LLM_MODEL_ID="microsoft/DialoGPT-medium"
export RAG_VERBOSE="true"
```

## 🧩 Module Details

### Core Modules

- **`config.py`**: Configuration management using Pydantic
- **`pipeline.py`**: Main orchestration logic

### Processors

- **`document.py`**: PDF reading, text extraction, and chunking
- **`embedding.py`**: Text embedding creation using sentence transformers
- **`llm.py`**: Language model loading and text generation
- **`retrieval.py`**: Vector similarity search and result ranking

### Utilities

- **`device.py`**: GPU/CPU detection and model memory management
- **`logging.py`**: Structured logging setup
- **`text.py`**: Text processing and formatting utilities

## 🔧 Extending the Pipeline

### Adding New Document Types

1. Create a new processor in `processors/`
2. Implement the same interface as `DocumentProcessor`
3. Update the pipeline to use your new processor

### Using Different Models

**Embedding Models:**
- `all-mpnet-base-v2` (default, good balance)
- `all-MiniLM-L6-v2` (faster, smaller)
- `all-roberta-large-v1` (slower, better quality)

**Language Models:**
- `microsoft/DialoGPT-medium` (lighter, faster)
- `google/gemma-2b-it` (instruction-tuned, requires authentication)
- `facebook/opt-1.3b` (larger, more capable)

### Custom Retrieval Logic

Extend `RetrievalSystem` to implement:
- Different similarity metrics
- Hybrid search (keyword + semantic)
- Re-ranking strategies

## 📊 Performance Notes

**Memory Usage:**
- Embedding model: ~1-2GB GPU memory
- LLM (with quantization): ~2-4GB GPU memory
- Document embeddings: ~100MB per 1000 chunks

**Processing Speed:**
- PDF processing: ~1-2 pages/second
- Embedding creation: ~100-500 chunks/second (GPU)
- Query processing: ~1-3 seconds per query

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   ```bash
   python main.py --device cpu --interactive
   # or
   python main.py --no-quantization --interactive
   ```

2. **Model authentication required:**
   ```bash
   huggingface-cli login
   ```

3. **spaCy model missing:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **PDF not found:**
   ```bash
   python main.py --pdf /full/path/to/document.pdf --interactive
   ```

### Debugging

Enable debug logging:
```bash
python main.py --log-level DEBUG --interactive
```

Check log files:
```bash
tail -f rag_pipeline_*.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the modular architecture
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Based on original work by Daniel Bourke
- Uses HuggingFace Transformers and Sentence Transformers
- Built with Rich for beautiful terminal output
