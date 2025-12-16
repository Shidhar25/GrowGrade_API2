# rag_engine.py
import os
import time
from chromadb import PersistentClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

client = PersistentClient(path="embeddings_store")

collection = client.get_or_create_collection(
    name="pdf_chunks_gemini",  # New collection for new model dimensions
    metadata={"hnsw:space": "cosine"}
)

# Replace HuggingFace with Gemini Embeddings
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from .env file")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # Upgraded model
    google_api_key=os.getenv("GEMINI_API_KEY")
)


def safe_int(value):
    """ChromaDB does NOT allow None, so convert None → -1"""
    return int(value) if value is not None else -1


def index_pdf_chunks(pdf_id, chunks):
    ids = [f"{pdf_id}_{c['id']}" for c in chunks]
    texts = [c["text"] for c in chunks]

    metadatas = []
    for c in chunks:
        metadatas.append({
            "pdf_id": pdf_id,
            "start_page": safe_int(c["start_page"]),
            "end_page": safe_int(c["end_page"]),
        })

    # --- FIX FOR 429 ERRORS: BATCHING & RATE LIMITING ---
    embeddings = []
    batch_size = 5  # Reduced batch size for Free Tier stability
    
    print(f"Starting embedding for {len(texts)} chunks...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        try:
            # Embed the current batch
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            
            # Rate Limit Protection: Sleep between batches
            # 2.0 seconds delay to stay well under RPM limits
            time.sleep(2.0) 
            
        except Exception as e:
            print(f"Error embedding batch starting at index {i}: {e}")
            raise e

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def retrieve(query, top_k=5):
    # Generate query embedding using Gemini
    query_emb = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    docs = []
    if results and len(results["documents"]) > 0:
        for i in range(len(results["documents"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
    return docs