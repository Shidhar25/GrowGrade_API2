# main.py
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from pdf_processor import extract_pages_text, chunk_document
from rag_engine import index_pdf_chunks, retrieve
from quiz_generator import generate_quiz_from_text
from rag_engine import collection  # optional: to inspect

load_dotenv()

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "RAG + Quiz Service Running"}


@app.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...)):
    """
    Upload PDF, chunk, index into ChromaDB, and generate a 10-question quiz for the whole PDF.
    Returns: {pdf_id, chunk_count, quiz}
    """
    # save
    filename = file.filename
    pdf_id = str(uuid.uuid4())  # unique id for this PDF
    saved_name = f"{pdf_id}_{filename}"
    pdf_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # extract pages and chunk
    pages = extract_pages_text(pdf_path)
    if not pages or all(not p for _, p in pages):
        return JSONResponse(status_code=400, content={"error": "PDF contains no extractable text (maybe scanned images). Use OCR first."})

    chunks = chunk_document(pages)
    chunk_count = index_pdf_chunks(pdf_id, chunks)

    # generate full-document quiz (non-RAG)
    # use concatenated text or the largest chunk as source; prefer whole text but keep limit
    whole_text = "\n".join([c["text"] for c in chunks])
    quiz = generate_quiz_from_text(whole_text)

    return {"pdf_id": pdf_id, "filename": filename, "chunks_indexed": chunk_count, "quiz": quiz}


@app.post("/ask")
async def ask(query: str = Form(...), pdf_id: str = Form(None)):
    """
    RAG QA: query optionally constrained to a pdf_id, otherwise global.
    Returns: {query, context:[...], answer}
    """
    # Build retrieval query (if pdf_id provided filter by metadata)
    results = retrieve(query, top_k=5)

    # If pdf_id provided, filter results
    if pdf_id:
        results = [r for r in results if r["metadata"].get("pdf_id") == pdf_id]

    # assemble context
    context_text = "\n\n".join([f"Pages {r['metadata'].get('start_page')}-{r['metadata'].get('end_page')}: {r['text']}" for r in results])

    # call a simple LLM prompt (reuse quiz_generator's LLM? Could add QA-specific LLM)
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "google/gemma-2-27b-it"),
                     base_url=os.getenv("OPENAI_BASE_URL"),
                     temperature=0.2)

    prompt = f"""
You are an assistant that must answer the question using ONLY the provided CONTEXT. If the answer is not contained, say "I don't know from the provided document."

CONTEXT:
{context_text}

QUESTION:
{query}

Answer concisely and cite pages if possible.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    return {"query": query, "context_used": context_text, "answer": answer, "retrieved": results}
