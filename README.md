# Caregiver AI Assistant

A multilingual RAG (Retrieval Augmented Generation) powered caregiving assistant with voice capabilities. This application provides intelligent responses to queries about mental health, financial aid, stress management, time management, self-care activities, and communication tips using retrieved knowledge from a curated database.

## Features

- **Retrieval Augmented Generation (RAG)**: Combines LLM capabilities with semantic search to provide accurate, context-aware responses
- **Vector Search**: Uses FAISS for efficient similarity-based document retrieval
- **Multilingual Support**: Designed to support multiple languages (English, Arabic, and more)
- **Voice Capabilities**: Supports audio input and text-to-speech output
- **Web Interface**: Interactive web UI built with HTML, CSS, and JavaScript
- **FastAPI Backend**: Modern, high-performance Python web framework
- **Local LLM**: Uses Ollama with Llama 3.2 for privacy-preserving inference
- **CORS Support**: Ready for frontend integration with CORS middleware configured

## Project Structure

```
├── main.py                 # FastAPI application and API endpoints
├── ingest.py              # Vector database creation and document ingestion
├── config.py              # Configuration settings
├── data/                  # Knowledge base documents
│   ├── communication_tips.txt
│   ├── financial_aid.txt
│   ├── mental_health_resources.txt
│   ├── self_care_activities.txt
│   ├── stress_management.txt
│   └── time_management.txt
├── faiss_index/           # FAISS vector database
│   └── index.faiss
├── templates/             # Jinja2 HTML templates
│   └── index.html
└── static/                # Frontend assets
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

## Requirements

- Python 3.8+
- FastAPI
- LangChain
- FAISS
- HuggingFace Embeddings
- ChatOllama (Ollama service running)
- Jinja2

## Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and run Ollama**
   - Download Ollama from [ollama.ai](https://ollama.ai)
   - Run Ollama service
   - Pull the required model:
     ```bash
     ollama pull llama3.2
     ```

5. **Prepare the knowledge base** (optional)
   - Add or modify `.txt` files in the `data/` folder with your knowledge base content

6. **Create the FAISS vector database**
   ```bash
   python ingest.py
   ```

## Configuration

Edit `config.py` to customize settings:

- `DB_FAISS_PATH`: Path to FAISS index (default: `faiss_index`)
- `EMBEDDING_MODEL`: HuggingFace embedding model (default: `BAAI/bge-small-en-v1.5`)
- `OLLAMA_MODEL`: Ollama model to use (default: `llama3.2`)
- `DATA_PATH`: Directory containing knowledge base documents (default: `data`)
- `CHUNK_SIZE`: Document chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `HOST`: Server host (default: `localhost`)
- `PORT`: Server port (default: 5000)
- `MAX_UPLOAD_SIZE`: Maximum file upload size (default: 10MB)
- `ALLOWED_AUDIO_EXTENSIONS`: Supported audio formats for voice input

## Usage

1. **Start the application**
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:5000`

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - Type your query in the chatbot interface
   - The assistant will retrieve relevant documents and generate a response

3. **API Endpoints**
   - Documentation available at `http://localhost:5000/docs` (Swagger UI)
   - ReDoc documentation at `http://localhost:5000/redoc`

## How It Works

1. **Document Ingestion** (`ingest.py`):
   - Loads text documents from the `data/` folder
   - Cleans the text by removing unwanted characters
   - Splits documents into chunks with configurable size and overlap
   - Creates embeddings using HuggingFace models
   - Stores embeddings in a FAISS vector database

2. **Query Processing** (`main.py`):
   - User submits a query through the web interface or API
   - Query is embedded using the same embedding model
   - FAISS retrieves the most relevant document chunks
   - Retrieved chunks and user query are passed to the Ollama LLM
   - LLM generates a contextual response based on the retrieved documents
   - Response is returned to the user

## Adding Knowledge

To add more knowledge to the assistant:

1. Create a new `.txt` file in the `data/` folder
2. Add your content to the file
3. Run `python ingest.py` to rebuild the vector database
4. Restart the application

## Development

### Rebuilding the FAISS Index
```bash
python ingest.py
```

### Running in Development Mode
```bash
python main.py
```

### Accessing API Documentation
After starting the server, visit:
- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

## Performance Notes

- Initial startup may take time as models are loaded from HuggingFace and Ollama
- The application caches models globally to avoid reload times for subsequent requests
- FAISS provides efficient similarity search even with large document collections

## Troubleshooting

- **Documents not loading**: Ensure `.txt` files exist in the `data/` folder and are in plain text format
- **Ollama connection error**: Make sure Ollama service is running (`ollama serve`)
- **Model download errors**: Check internet connection and available disk space
- **Port already in use**: Change `PORT` in `config.py`

## Future Enhancements

- Multi-language support with language detection
- Voice input and output integration
- Document upload functionality
- User session management
- Response rating and feedback collection
- Analytics and usage tracking

## Support

For issues or questions, please create an issue in the project repository.
