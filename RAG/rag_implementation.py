import json
import os
import sys
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARSED_JSON_PATH = os.path.join(BASE_DIR, "../Jupyter file/ans_parsed.json")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "faiss_index")

def load_parsed_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads the parsed notebook data from JSON."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_documents(parsed_blocks: List[Dict[str, Any]]) -> List[Document]:
    """
    Converts parsed blocks into LangChain Documents.
    
    Strategy:
    - CodeBlock: Content = Summary + "\n\nCode:\n" + Content. Metadata = type, outputs.
    - MarkdownBlock: Content = Content. Metadata = type.
    """
    documents = []
    for block in parsed_blocks:
        cell_type = block.get("cell_type")
        content = block.get("content", "")
        metadata = block.get("metadata", {})
        
        # Ensure metadata is a flat dict for compatibility with some vector stores
        # though FAISS handles dicts reasonably well, keeping it simple is safer.
        doc_metadata = {"cell_type": cell_type}
        
        if cell_type == "code":
            summary = block.get("summary", "")
            outputs = block.get("outputs", [])
            
            # Construct rich text for embedding
            # We prioritize the summary as it contains the semantic meaning
            text_to_embed = f"Summary: {summary}\n\nCode:\n{content}"
            
            # Add output info to metadata (simplified)
            doc_metadata["has_outputs"] = len(outputs) > 0
            doc_metadata["summary"] = summary # Store summary in metadata for retrieval display
            
            documents.append(Document(page_content=text_to_embed, metadata=doc_metadata))
            
        elif cell_type == "markdown":
            # Markdown is already text
            documents.append(Document(page_content=content, metadata=doc_metadata))
            
    return documents

def build_rag_database(json_path: str = PARSED_JSON_PATH, db_path: str = VECTOR_DB_PATH):
    """Builds and saves the FAISS vector database."""
    print(f"Loading data from {json_path}...")
    data = load_parsed_data(json_path)
    if not data:
        return

    print("Creating documents...")
    docs = create_documents(data)
    print(f"Created {len(docs)} documents.")

    print("Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    # This will download the model locally if not present
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vector_db = FAISS.from_documents(docs, embeddings)

    print(f"Saving index to {db_path}...")
    vector_db.save_local(db_path)
    print("Done!")

def query_rag(query: str, db_path: str = VECTOR_DB_PATH, k: int = 3):
    """Queries the existing FAISS database."""
    if not os.path.exists(db_path):
        print(f"Index not found at {db_path}. Please build it first.")
        return

    print("Loading embeddings and index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    print(f"\nQuery: {query}")
    print("-" * 40)
    
    results = vector_db.similarity_search(query, k=k)
    
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Type: {doc.metadata.get('cell_type')}")
        # Print a snippet of the content
        content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        print(f"Content:\n{content_preview}")
        print("-" * 20)

if __name__ == "__main__":
    # Simple CLI
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "build":
            build_rag_database()
        elif cmd == "query" and len(sys.argv) > 2:
            query_text = " ".join(sys.argv[2:])
            query_rag(query_text)
        else:
            print("Usage:")
            print("  python rag_implementation.py build")
            print("  python rag_implementation.py query 'Your question here'")
    else:
        # Default behavior if no args: build then test query
        build_rag_database()
        print("\nRunning test query...")
        query_rag("How is the image loaded?")
