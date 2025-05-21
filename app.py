import os
import json # Needed for storing sources as JSON
from datetime import datetime # Needed for timestamps
from bson.objectid import ObjectId # Needed for MongoDB ObjectIds
from flask import Flask, request, jsonify, render_template, session, redirect, url_for # Added session, redirect, url_for
from flask_pymongo import PyMongo # Added Flask-PyMongo

# ✅ RAG imports (keeping existing)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Dict, Any, Optional, List # Added List

# --- Configuration ---
FAISS_INDEX_DIR = "./faiss_index_legal"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral" # <--- *** ENSURE THIS MODEL IS AVAILABLE IN OLLAMA ***

# --- Flask App Setup ---
app = Flask(__name__)

# --- Flask Session Configuration ---
# Needed for storing user session data (like user_id)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_for_dev") # CHANGE THIS IN PRODUCTION!
# In development, you can just use 'super_secret_key_for_dev'. For production, use a random, long string from environment variables.

# --- MongoDB Configuration ---
# Assumes MongoDB is running on localhost:27017
# We'll use a database named 'legal_rag_db'
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/legal_rag_db")
app.config["MONGO_URI"] = MONGO_URI

# --- Initialize Flask-PyMongo ---
mongo = PyMongo(app)

# --- Global Variables for RAG Components ---
embedding: Optional[HuggingFaceEmbeddings] = None
vectorstore: Optional[FAISS] = None
rag_chain: Optional[RetrievalQA] = None
# ...existing code...

prompt_template = """You are a professional legal assistant powered by AI. Your role is to:
1. Provide clear explanations of legal concepts and procedures
2. Reference relevant laws, regulations, and precedents from the provided context
3. Help users understand their legal rights and obligations
4. Explain legal terminology in plain language

Important Guidelines:
- Always base responses on the provided context documents
- Cite specific sources, including document names and page numbers when available
- Clearly state when information is not available in the context
- Maintain a professional, clear, and objective tone



Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# ...existing code...
# --- Initialization Function for RAG ---
def initialize_rag_components():
    """Loads the embedding model, vector store, and LLM."""
    global embedding, vectorstore, rag_chain

    print("Initializing RAG components...")

    # 1. Initialize Embedding Model
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        # Allow server to start even if RAG init fails, but chat will not work
        rag_init_failed = True
        print("RAG components WILL NOT be available.")
        return False

    # 2. Load FAISS Index
    if not os.path.exists(FAISS_INDEX_DIR):
        print(f"FAISS index directory not found at {FAISS_INDEX_DIR}. RAG will not work.")
        # Allow server to start even if RAG init fails, but chat will not work
        rag_init_failed = True
        print("RAG components WILL NOT be available.")
        return False

    try:
        print(f"Loading FAISS index from {FAISS_INDEX_DIR}")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embedding,
            allow_dangerous_deserialization=True # Adjust based on your needs and security context
        )
        print("FAISS index loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        # Allow server to start even if RAG init fails, but chat will not work
        rag_init_failed = True
        print("RAG components WILL NOT be available.")
        return False


    # ✅ 3. Initialize LLM (Using Ollama)
    try:
        print(f"Loading Ollama model: {OLLAMA_MODEL_NAME}")
        llm = Ollama(model=OLLAMA_MODEL_NAME)
        # Optional: Add a quick test call to check if Ollama is reachable
        try:
            llm.invoke("Hi", config={"max_tokens": 5})
            print("Ollama model loaded and accessible.")
        except Exception as ollama_check_e:
             print(f"Warning: Ollama model '{OLLAMA_MODEL_NAME}' might not be running or accessible: {ollama_check_e}")
             print("Ensure Ollama is running and the model is pulled (`ollama pull your_model_name`).")
             # Decide if you want to fail initialization here or just warn.
             # For this example, we warn but let it proceed, first chat will fail if LLM is down.
             # return False # Uncomment to fail initialization if Ollama check fails

    except Exception as e:
        print(f"Error loading Ollama model '{OLLAMA_MODEL_NAME}': {e}")
        print("Ensure Ollama is installed, running, and the model is downloaded (`ollama pull your_model_name`).")
        # Allow server to start even if RAG init fails, but chat will not work
        rag_init_failed = True
        print("RAG components WILL NOT be available.")
        return False


    # 4. Create the RetrievalQA Chain
    try:
        print("Creating RetrievalQA chain.")
        retriever = vectorstore.as_retriever()

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("RetrievalQA chain created successfully.")
    except Exception as e:
        print(f"Error creating RetrievalQA chain: {e}")
        # Allow server to start even if RAG init fails, but chat will not work
        rag_init_failed = True
        print("RAG components WILL NOT be available.")
        return False


    print("RAG components initialized successfully.")
    return True # Initialization succeeded

# --- Authentication Helper ---
def login_required(f):
    """Decorator to protect routes."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user_id is in the session
        if 'user_id' not in session:
            # For API endpoints, return JSON error
            if request.path.startswith('/api/') or request.path == '/chat': # Adjust prefix if needed
                 return jsonify({"error": "Authentication required"}), 401
            # For UI routes, redirect to login page
            return redirect(url_for('index')) # Redirect to index which shows login
        return f(*args, **kwargs)
    return decorated_function


# --- Flask Route for the UI ---
@app.route('/')
def index():
    """Renders the main chat interface HTML page."""
    # If user is already logged in, perhaps redirect to the chat part or just show the page
    if 'user_id' in session:
         # You might want to render a different template or pass a flag
         # to the frontend to show the chat interface immediately.
         # For now, render index.html which chat.js handles based on login status.
         pass
    return render_template('index.html')

# --- Flask Route for Login ---
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # ⚠️ SECURITY WARNING: Storing and comparing plaintext passwords is NOT secure.
    print(f"⚠️ SECURITY WARNING: Attempting login for {email} with plaintext password.")
    user = mongo.db.users.find_one({"email": email, "password": password}) # ⚠️ Plaintext comparison

    if user:
        # Login successful
        # Store user ID in session
        session['user_id'] = str(user['_id']) # Store ObjectId as string
        print(f"User {email} logged in successfully.")
        # You might return user details here, but avoid sending sensitive data
        return jsonify({"message": "Login successful", "user_id": str(user['_id'])}), 200
    else:
        # Login failed
        print(f"Login failed for {email}.")
        return jsonify({"error": "Invalid email or password"}), 401

# --- Flask Route for Logout ---
@app.route('/logout', methods=['POST'])
@login_required # Ensure user is logged in to log out (optional, but good practice)
def logout():
    session.pop('user_id', None) # Remove user ID from session
    print("User logged out.")
    return jsonify({"message": "Logout successful"}), 200

# --- Flask Route to check Login Status (Useful for Frontend) ---
@app.route('/status', methods=['GET'])
def status():
     if 'user_id' in session:
          # Optionally fetch user details from DB using session['user_id']
          # user = mongo.db.users.find_one({"_id": ObjectId(session['user_id'])})
          # if user:
          #    return jsonify({"is_authenticated": True, "user_email": user.get("email", "N/A")}), 200
          # else:
          #    session.pop('user_id', None) # Clear session if user not found (DB issue?)
          #    return jsonify({"is_authenticated": False}), 200
          return jsonify({"is_authenticated": True}), 200 # Simple check
     else:
          return jsonify({"is_authenticated": False}), 200


# --- Flask Route for Registering a User (Optional, basic example) ---
# You could add this if you want users to create accounts
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password') # ⚠️ This is the plaintext password

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # Check if user already exists
    if mongo.db.users.find_one({"email": email}):
        return jsonify({"error": "User with this email already exists"}), 409 # 409 Conflict

    # ⚠️ SECURITY WARNING: Storing plaintext password
    print(f"⚠️ SECURITY WARNING: Registering user {email} with plaintext password.")
    try:
        user_doc = {
            "email": email,
            "password": password, # ⚠️ Storing plaintext
            "created_at": datetime.utcnow()
        }
        result = mongo.db.users.insert_one(user_doc)
        return jsonify({"message": "User registered successfully", "user_id": str(result.inserted_id)}), 201 # 201 Created
    except Exception as e:
        print(f"Error during user registration: {e}")
        return jsonify({"error": "Failed to register user"}), 500


# --- Flask Route for Starting a New Conversation ---
@app.route('/conversations/new', methods=['POST'])
@login_required
def start_new_conversation():
     user_id = session['user_id']
     try:
          # Create a new conversation document
          conversation_doc = {
               "user_id": ObjectId(user_id), # Store user_id as ObjectId
               "title": "New Chat", # Default title, can be updated later
               "created_at": datetime.utcnow(),
               "messages": [] # Embed messages directly in the conversation document
          }
          result = mongo.db.conversations.insert_one(conversation_doc)
          conversation_id = str(result.inserted_id)
          print(f"New conversation started for user {user_id}: {conversation_id}")
          return jsonify({"conversation_id": conversation_id}), 201
     except Exception as e:
          print(f"Error starting new conversation for user {user_id}: {e}")
          return jsonify({"error": "Failed to start new conversation"}), 500


# --- Flask Route for Getting User's Conversations ---
@app.route('/conversations', methods=['GET'])
@login_required
def get_conversations():
     user_id = session['user_id']
     try:
          # Find all conversations for the user, sorted by creation date
          conversations = mongo.db.conversations.find(
               {"user_id": ObjectId(user_id)},
               {"messages": 0} # Don't retrieve messages in this list view
          ).sort("created_at", -1) # Sort descending by created_at

          # Format for frontend
          conv_list = []
          for conv in conversations:
               conv_list.append({
                   "id": str(conv['_id']),
                   "title": conv.get("title", "Unnamed Chat"),
                   "created_at": conv['created_at'].isoformat() # Use ISO format for date
               })
          print(f"Retrieved {len(conv_list)} conversations for user {user_id}")
          return jsonify(conv_list), 200
     except Exception as e:
          print(f"Error retrieving conversations for user {user_id}: {e}")
          return jsonify({"error": "Failed to retrieve conversations"}), 500


# --- Flask Route for Getting Messages of a Conversation ---
@app.route('/conversations/<conversation_id>/messages', methods=['GET'])
@login_required
def get_messages(conversation_id):
    user_id = session['user_id']
    try:
        # Ensure the conversation ID is valid
        if not ObjectId.is_valid(conversation_id):
             return jsonify({"error": "Invalid conversation ID"}), 400

        # Find the conversation and ensure it belongs to the current user
        conversation = mongo.db.conversations.find_one(
            {"_id": ObjectId(conversation_id), "user_id": ObjectId(user_id)}
        )

        if not conversation:
            return jsonify({"error": "Conversation not found or does not belong to user"}), 404

        # Format messages for frontend
        messages_list = []
        # Assuming messages are embedded as an array in the conversation document
        for msg in conversation.get('messages', []):
            messages_list.append({
                "sender": msg.get("sender"),
                "text": msg.get("text"),
                "timestamp": msg.get("timestamp", datetime.utcnow()).isoformat(), # Use ISO format
                "sources": msg.get("sources", []) # Include sources for bot messages
            })

        print(f"Retrieved {len(messages_list)} messages for conversation {conversation_id}")
        # Include conversation title for frontend
        return jsonify({"title": conversation.get("title", "Unnamed Chat"), "messages": messages_list}), 200

    except Exception as e:
        print(f"Error retrieving messages for conversation {conversation_id}: {e}")
        return jsonify({"error": "Failed to retrieve messages"}), 500


# --- Flask Route for the Chat API ---
@app.route('/chat', methods=['POST'])
@login_required # Protect the chat endpoint
def chat():
    # Check if RAG components were successfully initialized on startup
    if rag_chain is None:
        print("RAG components not initialized. Returning 500 error.")
        return jsonify({"error": "RAG components not initialized. Server might be starting or failed to load resources."}), 500

    user_id = session['user_id']
    data: Dict[str, Any] = request.json
    query: Optional[str] = data.get('query')
    # Get conversation ID from the frontend request
    conversation_id_str: Optional[str] = data.get('conversation_id')


    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Invalid or empty 'query' provided in the request body."}), 400

    print(f"\nReceived query from user {user_id}: '{query}' (Conversation ID: {conversation_id_str})")

    try:
        # --- Chat History & Conversation Management ---
        conversation_id = None
        if conversation_id_str and ObjectId.is_valid(conversation_id_str):
            conversation_id = ObjectId(conversation_id_str)
            # Verify conversation belongs to the user (optional, but good practice)
            conversation = mongo.db.conversations.find_one({"_id": conversation_id, "user_id": ObjectId(user_id)})
            if not conversation:
                 print(f"Attempted to use invalid or mismatched conversation_id: {conversation_id_str} for user {user_id}")
                 return jsonify({"error": "Invalid or mismatched conversation ID"}), 400
        else:
            # This is the first message of a new conversation sequence from the frontend
            print(f"Starting new conversation sequence for user {user_id}")
            new_conv_doc = {
                "user_id": ObjectId(user_id),
                # Set a placeholder title initially, update after RAG response or first few messages
                "title": shorten_text(query, 50), # Simple title from first query
                "created_at": datetime.utcnow(),
                "messages": [] # Will embed messages
            }
            insert_result = mongo.db.conversations.insert_one(new_conv_doc)
            conversation_id = insert_result.inserted_id
            conversation_id_str = str(conversation_id) # Update string ID to return


        # Save user message to MongoDB
        user_message = {
            "sender": "user",
            "text": query,
            "timestamp": datetime.utcnow(),
            "sources": [] # User messages don't have sources
        }
        # Use $push to add message to the embedded array
        mongo.db.conversations.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": user_message}}
        )


        # --- RAG Processing ---
        response = rag_chain.invoke({"query": query})

        answer = response.get('result', 'Could not generate an answer.')
        source_docs = response.get('source_documents', [])

        # Format sources for the response and for storage
        formatted_sources = []
        raw_sources_for_storage = [] # Store a simplified version for the database
        for doc in source_docs:
            meta = doc.metadata
            snippet_length = 200
            content_snippet = doc.page_content[:snippet_length] + ('...' if len(doc.page_content) > snippet_length else '')

            display_source = f"Source: {meta.get('source', 'N/A')}"
            if 'page' in meta:
                display_source += f", Page: {meta['page']}"
                if 'total_pages' in meta:
                     display_source += f"/{meta['total_pages']}"
            if 'topic' in meta:
                 display_source += f", Topic: {meta['topic']}"
            if 'audience' in meta:
                 display_source += f", Audience: {meta['audience']}"
            # Include document ID if available - helps trace back to original chunks if needed
            doc_id = meta.get('doc_id', meta.get('source', 'N/A') + '_' + str(meta.get('page', 'N/A'))) # Fallback ID
            display_source += f" (Doc ID: {doc_id})"


            formatted_sources.append({
                "display": display_source,
                "content_snippet": content_snippet,
                # Optionally include full metadata if needed by frontend
                # "metadata": meta
            })

            # Store a simplified version in DB
            raw_sources_for_storage.append({
                 "source": meta.get('source', 'N/A'),
                 "page": meta.get('page'), # Allow None
                 "doc_id": doc_id,
                 "snippet": content_snippet # Store snippet
                 # Avoid storing full page_content unless necessary due to size
            })


        # Save bot message to MongoDB
        bot_message = {
            "sender": "bot",
            "text": answer,
            "timestamp": datetime.utcnow(),
            "sources": raw_sources_for_storage # Store simplified sources
        }
        mongo.db.conversations.update_one(
            {"_id": conversation_id},
            {"$push": {"messages": bot_message}}
        )

        print(f"Successfully processed query for conversation {conversation_id_str}.")
        return jsonify({
            "response": answer,
            "sources": formatted_sources,
            "conversation_id": conversation_id_str # Return the conversation ID
        })

    except Exception as e:
        print(f"An error occurred during processing query '{query}' for user {user_id}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        # Attempt to save an error message to the chat history
        if conversation_id:
             error_message = {
                 "sender": "bot",
                 "text": "Sorry, an error occurred while processing your request. Please try again.",
                 "timestamp": datetime.utcnow(),
                 "sources": []
             }
             try:
                 mongo.db.conversations.update_one(
                    {"_id": conversation_id},
                    {"$push": {"messages": error_message}}
                 )
             except Exception as db_e:
                  print(f"Failed to save error message to DB: {db_e}")

        return jsonify({"error": "An internal error occurred while processing your query.", "details": str(e)}), 500

# Add this new route for web search
@app.route('/web_search', methods=['POST'])
@login_required
def web_search():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({
            'error': 'No query provided'
        }), 400
    
    try:
        # Here you would implement your web search logic
        # For now, returning a placeholder response
        return jsonify({
            'response': f"Web search results for: {query}\n(Web search implementation pending)",
            'sources': [
                {
                    'display': 'Web Search',
                    'content_snippet': 'This is a placeholder for web search results.'
                }
            ]
        })
        
    except Exception as e:
        print(f"Error in web search: {e}")
        return jsonify({
            'error': 'Failed to perform web search',
            'details': str(e)
        }), 500
# --- Utility Functions ---
def shorten_text(text, max_length):
    """Helper to shorten text for conversation titles."""
    if not text:
        return "Unnamed Chat"
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].strip() + "..."


# --- Entry Point ---
if __name__ == '__main__':
    print("--- Starting Server Initialization ---")

    # Optional: Basic user creation on startup for easy testing if no users exist
    # ⚠️ WARNING: This creates a default user with a plaintext password every time!
    # Remove or modify this for persistent user management.
    # try:
    #     if mongo.db.users.count_documents({}) == 0:
    #         print("No users found. Creating a default user (test@example.com/password123)...")
    #         default_user_doc = {
    #             "email": "test@example.com",
    #             "password": "password123", # ⚠️ Plaintext!
    #             "created_at": datetime.utcnow()
    #         }
    #         mongo.db.users.insert_one(default_user_doc)
    #         print("Default user created.")
    # except Exception as e:
    #      print(f"Error checking/creating default user: {e}")


    # Initialize RAG components when the Flask app starts
    # Server will run even if RAG fails, but chat won't work
    rag_initialized_successfully = initialize_rag_components()

    if not rag_initialized_successfully:
        print("\n--- WARNING: RAG components failed to initialize. Chat functionality will NOT work. ---")

    print("\n--- Starting Flask server ---")
    # Use debug=True for development, set to False for production
    # host='0.0.0.0' makes it accessible externally (use with caution)
    app.run(debug=True, host='0.0.0.0', port=5000)