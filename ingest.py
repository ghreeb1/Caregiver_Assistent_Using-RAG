# ingest.py
import os
import re
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_PATH, DB_FAISS_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def clean_text(text):
    """
    Removes unwanted characters like *, -, # from text.
    Add more symbols to the regex if needed.
    """
    return re.sub(r'[\*\-\#]', '', text)

def create_vector_db():
    # Ensure the data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory not found at {DATA_PATH}. Please create it and add .txt files.")
        return

    # Load documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found to process. Please add .txt files to the data/ directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Clean documents text
    print("Cleaning documents from unwanted symbols (*, -, #)...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Split documents into chunks
    print(f"Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Create embeddings
    print("Creating embeddings (this may take a while)...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create FAISS vector store
    print("Creating FAISS vector store...")
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the vector store to the specified directory
    vector_store.save_local(DB_FAISS_PATH)
    print(f"Vector database saved to {DB_FAISS_PATH}")
    print("Ingestion complete!")

if __name__ == "__main__":
    create_vector_db()
