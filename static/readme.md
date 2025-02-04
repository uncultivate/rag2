# RAG Chat Interface

## Setup

1. Clone the repository
2. Install dependencies:   ```bash
   pip install -r requirements.txt   ```
3. Copy `.env.example` to `.env`:   ```bash
   cp .env.example .env   ```
4. Update `.env` with your actual API keys and configuration
5. Run the application:   ```bash
   python app.py   ```

## Environment Variables

The following environment variables are required:

- `GROQ_KEY`: Your Groq API key
- `AZURE_SEARCH_SERVICE`: Your Azure Search Service endpoint