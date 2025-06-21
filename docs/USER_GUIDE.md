# 📖 User Guide
## Modular RAG Pipeline v2.0

### Getting Started

Welcome to the Modular RAG Pipeline! This guide will help you get up and running quickly with querying your PDF documents using AI.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd rag-pipeline

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Basic Usage
```bash
# Start with your PDF document
python main.py --pdf your_document.pdf --interactive

# Or use our sample document
python main.py --interactive
```

---

## 🎯 Core Concepts

### What is RAG?
**Retrieval Augmented Generation (RAG)** combines the power of information retrieval with language generation:

1. **📄 Document Processing**: Your PDF is split into searchable chunks
2. **🧠 Embedding Creation**: Each chunk gets a vector representation
3. **🔍 Retrieval**: Find the most relevant chunks for your query
4. **💬 Generation**: AI generates an answer based on relevant context

### How It Works
```
Your Question → Vector Search → Relevant Text → AI Answer
     ↓              ↓              ↓           ↓
"What is X?"  →  [0.1, 0.8...]  →  "X is..."  →  "Based on the document, X is..."
```

---

## 🎮 Usage Modes

### 1. Interactive Mode
Perfect for exploratory questioning and conversation-like interactions.

```bash
python main.py --interactive
```

**Features:**
- Real-time question answering
- Beautiful terminal interface
- Context-aware responses
- Easy to quit with `exit` or `quit`

**Example Session:**
```
💬 Enter your query: What are the main topics in this document?

📊 Search Results
┌─────────┬──────┬─────────────────────────────────────────────┐
│ Score   │ Page │ Content                                     │
├─────────┼──────┼─────────────────────────────────────────────┤
│ 0.8945  │  23  │ The document covers machine learning...     │
│ 0.8721  │  45  │ Key topics include neural networks...       │
└─────────┴──────┴─────────────────────────────────────────────┘

📝 Generated Answer
The document covers several main topics including machine learning 
fundamentals, neural network architectures, and practical applications...

💬 Enter your query: Tell me more about neural networks
```

### 2. Single Query Mode
Best for scripting and automation.

```bash
python main.py --query "What is the main conclusion?"
```

**Use Cases:**
- Batch processing
- Automated reports
- Quick fact extraction
- Integration with other tools

### 3. Sample Queries Mode
Great for testing and demonstration.

```bash
python main.py --sample-queries
```

**Includes queries like:**
- "What are the macronutrients?"
- "How do vitamins differ from minerals?"
- "What is the process of digestion?"

---

## ⚙️ Configuration

### Command Line Options

#### Basic Options
```bash
# Document settings
--pdf PATH              Path to PDF document
--pdf-url URL          Download PDF from URL

# Model settings  
--model MODEL_ID       HuggingFace model for LLM
--embedding-model NAME Embedding model name
--device {auto,cpu,cuda} Force device usage

# Output settings
--verbose              Enable detailed output
--log-level LEVEL      Set logging level
```

#### Advanced Options
```bash
# Performance tuning
--no-quantization      Disable 4-bit quantization
--embeddings PATH      Custom embeddings file path

# Configuration
--config CONFIG_FILE   Use JSON configuration file
```

### Configuration File
Create a `config.json` file for persistent settings:

```json
{
  "document": {
    "pdf_path": "research_paper.pdf",
    "min_token_length": 30,
    "num_sentence_chunk_size": 10
  },
  "models": {
    "embedding_model_name": "all-mpnet-base-v2",
    "llm_model_id": "microsoft/DialoGPT-medium",
    "use_quantization": true
  },
  "retrieval": {
    "n_resources_to_return": 5,
    "temperature": 0.7,
    "max_new_tokens": 512
  },
  "system": {
    "verbose": true,
    "log_level": "INFO"
  }
}
```

Then use it:
```bash
python main.py --config config.json --interactive
```

### Environment Variables
Set environment variables for system-wide configuration:

```bash
export RAG_PDF_PATH="./documents/default.pdf"
export RAG_LLM_MODEL_ID="microsoft/DialoGPT-medium"
export RAG_VERBOSE="true"
```

---

## 🎛️ Model Selection

### Embedding Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | Small | Fast | Good | Quick testing, limited resources |
| `all-mpnet-base-v2` | Medium | Moderate | Excellent | **Recommended default** |
| `all-roberta-large-v1` | Large | Slow | Best | High-quality applications |

```bash
# Fast and lightweight
python main.py --embedding-model all-MiniLM-L6-v2 --interactive

# Best quality (default)
python main.py --embedding-model all-mpnet-base-v2 --interactive

# Highest quality
python main.py --embedding-model all-roberta-large-v1 --interactive
```

### Language Models

| Model | Size | Memory | Quality | Authentication |
|-------|------|--------|---------|----------------|
| `microsoft/DialoGPT-medium` | 1.5B | 4GB | Good | ❌ None |
| `microsoft/DialoGPT-large` | 2.7B | 8GB | Better | ❌ None |
| `google/gemma-2b-it` | 2B | 6GB | Excellent | ✅ Required |

```bash
# Lightweight option
python main.py --model microsoft/DialoGPT-medium --interactive

# Better quality
python main.py --model microsoft/DialoGPT-large --interactive

# Best quality (requires HuggingFace login)
python main.py --model google/gemma-2b-it --interactive
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
Error: CUDA out of memory
```

**Solutions:**
```bash
# Use CPU instead
python main.py --device cpu --interactive

# Enable quantization (default)
python main.py --interactive

# Use smaller model
python main.py --model microsoft/DialoGPT-medium --interactive
```

#### 2. Model Authentication Required
```
Error: 401 Client Error. Cannot access gated repo
```

**Solution:**
```bash
# Login to HuggingFace
huggingface-cli login

# Or use public models
python main.py --model microsoft/DialoGPT-medium --interactive
```

#### 3. PDF Not Found
```
Error: PDF file not found
```

**Solutions:**
```bash
# Check file path
ls -la your_document.pdf

# Use absolute path
python main.py --pdf /full/path/to/document.pdf --interactive

# Download from URL
python main.py --pdf-url "https://example.com/doc.pdf" --pdf document.pdf --interactive
```

#### 4. spaCy Model Missing
```
Error: Can't find model 'en_core_web_sm'
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Performance Optimization

#### For Limited Memory
```bash
# Minimal memory configuration
python main.py \
  --model microsoft/DialoGPT-medium \
  --embedding-model all-MiniLM-L6-v2 \
  --device cpu \
  --interactive
```

#### For Best Performance
```bash
# High-performance configuration
python main.py \
  --model microsoft/DialoGPT-large \
  --embedding-model all-mpnet-base-v2 \
  --device cuda \
  --no-quantization \
  --interactive
```

#### For Best Quality
```bash
# Quality-focused configuration (requires auth)
python main.py \
  --model google/gemma-2b-it \
  --embedding-model all-roberta-large-v1 \
  --device cuda \
  --interactive
```

---

## 📊 Understanding Results

### Search Results Table
```
📊 Search Results
┌─────────┬──────┬─────────────────────────────────────────────┐
│ Score   │ Page │ Content                                     │
├─────────┼──────┼─────────────────────────────────────────────┤
│ 0.8945  │  23  │ The document covers machine learning...     │
│ 0.8721  │  45  │ Key topics include neural networks...       │
└─────────┴──────┴─────────────────────────────────────────────┘
```

- **Score**: Similarity score (0.0-1.0, higher is better)
- **Page**: Source page number in the PDF
- **Content**: Relevant text chunk from the document

### Quality Indicators

#### High Quality Response
- ✅ High similarity scores (>0.8)
- ✅ Multiple relevant sources
- ✅ Coherent, well-structured answer
- ✅ Cites specific information from the document

#### Lower Quality Response
- ⚠️ Low similarity scores (<0.6)
- ⚠️ Generic or vague answer
- ⚠️ Limited context from document
- ⚠️ May indicate question not covered in document

---

## 🎯 Best Practices

### Asking Good Questions

#### ✅ Effective Questions
```
❓ "What are the main symptoms of vitamin C deficiency?"
❓ "How does the paper define machine learning?"
❓ "What methodology was used in the study?"
❓ "What are the key findings regarding X?"
```

#### ❌ Less Effective Questions
```
❓ "Tell me everything"  (too broad)
❓ "What do you think?"  (asks for opinion)
❓ "Is this good?"       (subjective)
❓ "What about cats?"    (likely not in document)
```

### Document Preparation

#### ✅ Good Documents
- Clear, well-structured text
- Proper headings and sections
- High-quality PDF (not scanned images)
- Reasonable length (10-1000 pages)

#### ⚠️ Challenging Documents
- Scanned images without OCR
- Heavy formatting or tables
- Very short documents (<5 pages)
- Extremely long documents (>1000 pages)

### Performance Tips

1. **Start Small**: Test with shorter documents first
2. **Use GPU**: Significantly faster than CPU
3. **Cache Results**: Embeddings are saved and reused
4. **Adjust Settings**: Tune chunk size and retrieval count
5. **Monitor Memory**: Use quantization for large models

---

## 🆘 Getting Help

### Log Files
Check the log files for detailed error information:
```bash
tail -f rag_pipeline_*.log
```

### Verbose Mode
Enable verbose output for debugging:
```bash
python main.py --verbose --log-level DEBUG --interactive
```

### Community Support
- 📖 **Documentation**: Check our comprehensive docs
- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Discussions**: Join our community forum
- 📧 **Email**: Contact support team

### System Information
To report issues, include:
```bash
# System info
python --version
pip list | grep torch
nvidia-smi  # if using GPU

# Pipeline info
python main.py --help
```

Happy querying! 🚀
