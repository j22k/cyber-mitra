import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document # Need to import Document explicitly
from typing import List

# --- Configuration ---

# Directory containing your JSONL source files
SOURCE_DATA_DIR = "./data/final"

# Directory where the FAISS index will be saved (must match app.py)
FAISS_INDEX_DIR = "./faiss_index_legal"

# Embedding model name (must match app.py)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text Splitter Configuration
CHUNK_SIZE = 500  # Size of each text chunk
CHUNK_OVERLAP = 50 # Overlap between chunks to maintain context

# --- Document Loading Function (from JSONL files) ---

def load_documents_from_jsonl(directory_path: str) -> List[Document]:
    """
    Loads documents from multiple JSONL files in a directory.
    Assumes each line in a file is a JSON object like:
    {"id": "...", "text": "...", "metadata": {...}}
    Extracts 'text' as page_content and includes original 'metadata',
    'id', and 'source_file' in the final Document metadata.
    """
    documents = []
    if not os.path.exists(directory_path):
        print(f"Error: Source data directory not found at {directory_path}")
        return documents # Return empty list if directory doesn't exist

    print(f"Loading documents from {directory_path}...")
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        # Parse the JSON object from the line
                        item = json.loads(line)

                        # Extract text content (page_content)
                        # Use .get() with a default to avoid KeyError if 'text' is missing
                        text_content = item.get('text', '')
                        if not text_content:
                             print(f"Warning: Skipping line {line_num+1} in {filename} due to empty 'text'.")
                             continue

                        # Extract original metadata (use .get() with default empty dict)
                        original_metadata = item.get('metadata', {})

                        # Create new metadata for the Document object
                        # Include original metadata, the item's ID, and the source filename
                        combined_metadata = {
                            **original_metadata, # Merge original metadata
                            'doc_id': item.get('id'), # Add the item's ID from the line
                            'source_file': filename, # Add the source filename
                            'line_in_file': line_num + 1 # Optional: track line number
                        }
                        # Ensure non-serializable types are handled if necessary,
                        # though standard JSON primitives should be fine.

                        # Create a LangChain Document object
                        doc = Document(page_content=text_content, metadata=combined_metadata)
                        documents.append(doc)

                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON line {line_num+1} in {filename}: {e}")
                    except Exception as e:
                         print(f"Warning: Skipping line {line_num+1} in {filename} due to unexpected error: {e}")

    print(f"Finished loading. Total documents loaded: {len(documents)}")
    return documents

# --- Main Injection Process ---

def create_faiss_index():
    """
    Loads documents, splits them, creates embeddings, builds a FAISS index, and saves it.
    """
    # 1. Load documents from source directory
    loaded_documents = load_documents_from_jsonl(SOURCE_DATA_DIR)

    if not loaded_documents:
        print("No documents loaded. Aborting index creation.")
        return

    # 2. Initialize the text splitter
    print(f"\nInitializing text splitter (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len, # Use character length
        add_start_index=True # Optional: Add start index to metadata
    )

    # 3. Split documents into chunks
    print(f"Splitting {len(loaded_documents)} documents into chunks...")
    text_chunks = text_splitter.split_documents(loaded_documents)
    print(f"Created {len(text_chunks)} text chunks.")

    if not text_chunks:
        print("No text chunks created after splitting. Aborting index creation.")
        return

    # 4. Initialize the embedding model
    try:
        print(f"\nInitializing embedding model: {EMBEDDING_MODEL_NAME}...")
        # Ensure the model is downloaded locally if necessary for HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Aborting index creation.")
        return

    # 5. Create the FAISS vector store from chunks and embeddings
    try:
        print("Creating FAISS vector store from chunks and embeddings...")
        # This step generates embeddings for all chunks and builds the index
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        print("FAISS vector store created successfully.")
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        print("Aborting index creation.")
        return

    # 6. Save the FAISS vector store to disk
    try:
        print(f"\nSaving FAISS index to directory: {FAISS_INDEX_DIR}...")
        # Create the directory if it doesn't exist
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_DIR)
        print("FAISS index saved successfully.")
        print(f"\nIndex creation complete. The index is ready for use by app.py in {FAISS_INDEX_DIR}")

    except Exception as e:
        print(f"Error saving FAISS vector store: {e}")
        print("Index creation failed during save.")


# --- Entry Point ---
if __name__ == '__main__':
    create_faiss_index()