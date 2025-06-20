# Local RAG Pipeline

A robust implementation of a local Retrieval Augmented Generation (RAG) pipeline that enables you to query PDF documents using a combination of embedding models and Large Language Models (LLMs), all running on your local hardware.

## Features

- **Complete RAG Pipeline**: Document processing, embedding creation, retrieval system, and LLM response generation
- **Runs Locally**: Uses your own GPU/CPU for privacy and cost efficiency
- **Beautiful Terminal UI**: Rich, colorful terminal output with progress bars and formatted results
- **Error Handling**: Comprehensive error handling and logging
- **Interactive Mode**: Chat with your documents in an interactive terminal session
- **Customizable**: Configure embedding models, LLMs, and pipeline parameters
- **Efficiency**: Uses batched operations and GPU acceleration where available

## Requirements

- Python 3.8+
- NVIDIA GPU (recommended, but CPU mode is also available)
- For GPU acceleration: CUDA 11.7+ (to use Flash Attention 2, CUDA compute capability 8.0+ is required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-rag-pipeline.git
cd local-rag-pipeline
```

2. Install the required packages:
```bash
pip install -r requirements_script.txt
```

3. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
python local_rag.py --pdf your_document.pdf --interactive
```

### Command Line Arguments

- `--pdf PATH`: Path to the PDF document
- `--pdf-url URL`: URL to download PDF if not present locally
- `--embeddings PATH`: Path to save/load embeddings
- `--model MODEL_ID`: HuggingFace model ID for the LLM
- `--sample-queries`: Run with a set of sample queries
- `--interactive`: Start an interactive chat session
- `--query "your question"`: Run with a single query
- `--config PATH`: Path to a JSON configuration file

### Interactive Mode

Start an interactive session where you can chat with your document:

```bash
python local_rag.py --pdf your_document.pdf --interactive
```

### Sample Queries Mode

Run a set of predefined sample queries against your document:

```bash
python local_rag.py --pdf your_document.pdf --sample-queries
```

## Configuration

You can customize the pipeline by creating a JSON configuration file:

```json
{
  "pdf_path": "your_document.pdf",
  "pdf_download_url": "https://example.com/your_document.pdf",
  "min_token_length": 30,
  "num_sentence_chunk_size": 10,
  "embedding_model_name": "all-mpnet-base-v2",
  "llm_model_id": "google/gemma-2b-it",
  "use_quantization": true,
  "n_resources_to_return": 5,
  "temperature": 0.7,
  "max_new_tokens": 512
}
```

Then use it with:

```bash
python local_rag.py --config your_config.json
```

## Pipeline Workflow

1. **Document Processing**:
   - Load PDF document
   - Extract text and metadata
   - Split text into sentences and chunks

2. **Embedding Creation**:
   - Create numerical representations of text chunks
   - Store embeddings for later retrieval

3. **Retrieval System**:
   - Perform vector similarity search to find relevant text chunks
   - Return the most similar text chunks based on a query

4. **LLM Generation**:
   - Format a prompt with query and retrieved context
   - Generate a human-readable response using the LLM

## Example

```bash
# Download a sample document and start interactive mode
python local_rag.py --pdf-url "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf" --pdf nutrition.pdf --interactive
```

Query example: "What are the symptoms of vitamin C deficiency?"

## Performance Considerations

- GPU acceleration is recommended for reasonable performance
- For LLM models, 4-bit quantization is used by default to reduce memory requirements
- Batched embedding creation is used for better GPU utilization

## Customizing Models

### Embedding Models

The default embedding model is `all-mpnet-base-v2`, but you can use any model from the sentence-transformers library.

### LLM Models

The default LLM is `google/gemma-2b-it`, but you can use many other models like:
- `google/gemma-7b-it`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `TheBloke/Llama-2-7B-Chat-GGUF`

LLM selection depends on your available GPU memory.

## Logging

Logs are saved to `rag_pipeline_{timestamp}.log` with rotation at 10MB.

## License

MIT
