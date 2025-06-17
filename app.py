from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required API keys. Please check your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

try:
    # Initialize components with error handling
    logger.info("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()
    
    index_name = "arogya-sahayak"
    
    logger.info("Connecting to Pinecone...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
    
    logger.info("Initializing LLM...")
    llm = OpenAI(temperature=0.4, max_tokens=500)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    logger.info("Application initialized successfully!")
    
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise

@app.route("/")
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return jsonify({"error": "Template not found"}), 500

@app.route("/get", methods=["POST"])  # Only POST method
def chat():
    try:
        # Validate input
        user_message = request.form.get("msg")
        if not user_message or not user_message.strip():
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"User query: {user_message}")
        
        # Process the query
        response = rag_chain.invoke({"input": user_message.strip()})
        
        # Validate response
        if not response or "answer" not in response:
            logger.error("Invalid response from RAG chain")
            return jsonify({"error": "Failed to generate response"}), 500
        
        answer = response["answer"]
        logger.info(f"Generated response: {answer}")
        
        return str(answer)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Don't use debug=True in production
    app.run(host="0.0.0.0", port=8080, debug=False)
