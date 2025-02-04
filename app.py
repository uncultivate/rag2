from flask import Flask, render_template, request, jsonify, session
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from groq import Groq
import os
from dotenv import load_dotenv
from markupsafe import escape
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management


# Initialize your clients and configurations
AZURE_SEARCH_SERVICE = os.getenv('AZURE_SEARCH_SERVICE')
# Get API key from environment variable
GROQ_KEY = os.getenv('GROQ_KEY')
INDEX_NAME = "py-rag-tutorial-idx"

credential = DefaultAzureCredential()
groq_client = Groq(api_key=GROQ_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE,
    index_name=INDEX_NAME,
    credential=credential
)

# Modify the prompt to include chat history
GROUNDED_PROMPT = """
You are an AI assistant that helps users learn from the information found in the source material.
Answer the query using only the sources provided below and considering the chat history provided.

Previous conversation:
{chat_history}

Current query: {query}

Sources:
{sources}

Answer ONLY with the facts listed in the list of sources above. Cite your source when you answer the question.
If there isn't enough information in the sources, say you don't know.
Do not generate answers that don't use the sources above.
"""

@app.route('/')
def home():
    if 'chat_history' not in session:
        session['chat_history'] = []
    # Escape any HTML in the chat history
    safe_history = [{
        'query': escape(msg['query']),
        'response': escape(msg['response'])
    } for msg in session['chat_history']]
    return render_template('index.html', chat_history=safe_history)

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.json.get('query')
        
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Format chat history for the prompt
        formatted_history = "\n".join([
            f"User: {msg['query']}\nAssistant: {msg['response']}" 
            for msg in chat_history
        ])

        # Set up vector query
        vector_query = VectorizableTextQuery(
            text=user_query, 
            k_nearest_neighbors=50, 
            fields="text_vector"
        )

        # Search results
        search_results = search_client.search(
            query_type="semantic",
            semantic_configuration_name="my-semantic-config",
            scoring_profile="my-scoring-profile",
            scoring_parameters=["tags-bimberi, incident"],
            search_text=user_query,
            vector_queries=[vector_query],
            select="title, chunk, locations",
            top=5,
        )

        # Format sources
        sources_formatted = "=================\n".join([
            f'TITLE: {document["title"]}, CONTENT: {document["chunk"]}, LOCATIONS: {document["locations"]}' 
            for document in search_results
        ])

        # Get response from Groq
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": GROUNDED_PROMPT.format(
                        query=user_query,
                        chat_history=formatted_history,
                        sources=sources_formatted
                    )
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7
        )

        assistant_response = response.choices[0].message.content

        # Update chat history
        chat_history.append({
            'query': user_query,
            'response': assistant_response
        })
        session['chat_history'] = chat_history

        return jsonify({
            'query': user_query,
            'response': assistant_response
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a route to clear chat history
@app.route('/clear', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    return jsonify({'status': 'success'})

# Add after your app initialization
if not app.debug:
    logger = logging.getLogger(__name__)
    logger.addHandler(AzureLogHandler())
    logger.setLevel(logging.INFO)

# Update the main section
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))