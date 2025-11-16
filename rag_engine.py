# rag_engine.py
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings

client = PersistentClient(path="embeddings_store")

collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

    embeddings = embedding_model.embed_documents(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def retrieve(query, top_k=5):
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
