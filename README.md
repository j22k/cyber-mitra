# Cyber Mitra - AI Legal Assistant

Cyber Mitra is an AI-powered legal assistant that helps users understand legal procedures and information through natural conversation. Built with a sophisticated RAG (Retrieval Augmented Generation) architecture, it combines a modern web interface with powerful legal information retrieval capabilities.

![AI Legal Assistant Screenshot](static/assets/images/logo.svg)

## Features

### User Interface
- **Professional Legal Theme**: Clean, sophisticated design with law-inspired color scheme
- **Interactive Chat**: Real-time messaging with typing indicators and message history
- **Web Search Integration**: Toggle between chat and web search modes
- **Document Upload**: Support for legal document analysis
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Dark Mode**: Toggle between light and dark themes
- **Message History**: View and manage previous conversations

### Technical Features
- **RAG Architecture**: Uses FAISS for efficient document retrieval
- **Local LLM Integration**: Runs on your hardware with Ollama using the Mistral model
- **Vector Search**: Semantic search using HuggingFace embeddings
- **MongoDB Storage**: Persistent storage for conversations and user data
- **Session Management**: Secure user authentication and session handling
- **Performance Monitoring**: Built-in evaluation tools for RAG performance

## Project Structure

```
ai_legal/
├── app.py                    # Main Flask application
├── Ingestion.py              # Document ingestion and indexing
├── evaluate_rag.py           # RAG evaluation tools
├── enhanced_evaluation.py    # Extended evaluation features
├── static/                   # Static assets
│   ├── assets/
│   │   ├── css/              # Stylesheets
│   │   ├── js/               # JavaScript files
│   │   └── images/           # UI assets
└── templates/                # HTML templates
    ├── index.html            # Main chat interface
    └── about.html            # About page
```

## Installation

### Prerequisites
- Python 3.8+
- MongoDB
- Ollama

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/ai_legal.git
cd ai_legal
```

### Step 2: Install required packages
```bash
pip install -r requirements.txt
```

### Step 3: Install and start MongoDB
Follow the [MongoDB Installation Guide](https://docs.mongodb.com/manual/installation/) for your operating system.

Start MongoDB:
```bash
mongod
```

### Step 4: Install and configure Ollama
Install Ollama following the instructions for your operating system from [Ollama's website](https://ollama.ai/):

```bash
curl https://ollama.ai/install.sh | sh
```

Pull the Mistral model:
```bash
ollama pull mistral
```

Start Ollama:
```bash
ollama serve
```

### Step 5: Set up environment variables
```bash
cp .env.example .env
```
Edit the `.env` file with your specific configuration settings.

### Step 6: Run the application
```bash
python app.py
```

### Step 7: Access the application
Open `http://localhost:5000` in your browser, create an account or use test credentials, and start chatting with the AI legal assistant.

## Development Guide

### Adding New Features
- Add routes in `app.py`
- Add frontend code in `static/assets/js/`
- Add styles in `static/assets/css/`
- Update templates in `templates/`

### RAG Components
- Document ingestion: Use `Ingestion.py`
- Evaluation: Use `evaluate_rag.py` and `enhanced_evaluation.py`
- Modify prompts in `app.py` under `prompt_template`

### Running Evaluations
```bash
# Run RAG evaluation
python evaluate_rag.py

# View detailed metrics
python enhanced_evaluation.py
```

## Configuration

Key configuration options in `.env`:
```
MONGO_URI=mongodb://localhost:27017/legal_rag_db
FLASK_SECRET_KEY=your_secret_key
FAISS_INDEX_DIR=./faiss_index_legal
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL_NAME=mistral
```

## Ollama Integration Details

Cyber Mitra uses Ollama to run the Mistral model locally on your hardware, providing:

- **Privacy**: All legal queries and responses remain on your system
- **No API Costs**: Avoid spending on commercial LLM APIs
- **Customization**: Fine-tune model parameters for legal domain
- **Offline Usage**: Full functionality without internet access once models are downloaded

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Security Notes
- **Authentication**: Currently uses basic authentication - enhance for production
- **Passwords**: Implement proper password hashing before production use
- **Session Management**: Uses Flask sessions - consider additional security measures
- **API Protection**: Add rate limiting and additional security headers

## License
MIT License

## Credits
- Developed by Boehm Tech
- Icons by Font Awesome
- UI Components by Boehm Tech Design Team

## Support
For support, email support@boehmtech.com or open an issue in this repository.