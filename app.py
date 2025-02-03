from flask import Flask, render_template, request, jsonify
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize your clients and configurations
AZURE_SEARCH_SERVICE = "https://rag930.search.windows.net"
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

GROUNDED_PROMPT = """
You are an AI assistant that helps users learn from the information found in the source material.
Answer the query using only the sources provided below.
Use bullets if the answer has multiple points.
If the answer is longer than 3 sentences, provide a summary.
Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Query: {query}
Sources:\n{sources}
"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    
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
                    sources=sources_formatted
                )
            }
        ],
        model="llama3-8b-8192",
        temperature=0.7
    )

    return jsonify({
        'query': user_query,
        'response': response.choices[0].message.content
    })

if __name__ == '__main__':
    app.run(debug=True)