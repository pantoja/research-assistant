# Research Assistant

A scientific research assistant that uses RAG (Retrieval-Augmented Generation) to answer questions based on your PDF papers. The system indexes your papers, searches for relevant content, and generates answers citing the sources.

## What is RAG?

RAG combines three steps:
1. **Retrieval**: Searches for relevant document chunks in a vector database
2. **Augmented**: Adds those chunks as context to the prompt
3. **Generation**: An LLM generates an answer based on that context

This allows the model to answer questions using your specific documents rather than just its general training.

## Features

- 📚 Index multiple PDF papers
- 🔍 Semantic search using embeddings
- 💡 Context-aware answers with source citations
- 🗨️ Interactive chat mode

## Requirements

- Python 3.8+
- Ollama with `llama3.1:8b` model

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and download the model:
```bash
ollama pull llama3.1:8b
```

## Usage

### 1. Index Your Papers

Place your PDF files in the `papers/` folder, then run:

```bash
python index_papers.py
```

### 2. Ask Questions

Start the interactive assistant:

```bash
python research_assistant.py
```

Example interaction:
```
You: What is photosynthesis?
🔍 Searching papers...

💡 Answer:
Photosynthesis is the process by which plants convert light energy into chemical energy...

📄 Sources:
  • plant_biology_paper.pdf
```

Type `exit` or `quit` to stop.

## Project Structure

```
.
├── papers/                  # Place your PDF files here
├── chroma_db/              # Vector database (auto-generated)
├── index_papers.py       # Script to index PDFs
├── research_assistant.py   # Interactive Q&A assistant
└── README.md
```

## Notes

- Re-run `index_papers.py` whenever you add new papers
- The embeddings model runs on Apple Silicon GPU (MPS) but can be changed to CPU
- ChromaDB persists to disk, so indexing is only needed once per document set
- Uses updated LangChain packages (`langchain-chroma`, `langchain-huggingface`, `langchain-ollama`) to avoid deprecation warnings
