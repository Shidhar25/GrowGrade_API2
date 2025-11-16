# pdf_processor.py
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_pages_text(pdf_path):
    """
    Returns list of (page_number, page_text).
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text.strip()))
    return pages


def chunk_document(pages, chunk_size=800, chunk_overlap=100):
    """
    Convert page texts into chunk objects with metadata.
    Returns list of dicts: {"id": idx, "text": ..., "start_page": x, "end_page": y}
    """
    # concatenate pages with page markers to preserve page boundaries
    full_text = ""
    page_markers = []
    for page_num, text in pages:
        marker = f"\n\n[PAGE {page_num}]\n"
        page_markers.append((len(full_text), page_num))
        full_text += marker + text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)

    # map each chunk back to page ranges by searching contained page markers
    chunk_objs = []
    for i, chunk in enumerate(chunks):
        # find pages covered by chunk by scanning for "[PAGE X]" tokens
        pages_in_chunk = []
        for page_num in range(1, len(pages)+1):
            token = f"[PAGE {page_num}]"
            if token in chunk:
                pages_in_chunk.append(page_num)
        start_page = pages_in_chunk[0] if pages_in_chunk else None
        end_page = pages_in_chunk[-1] if pages_in_chunk else None

        # remove page markers from chunk text for cleanliness
        cleaned = chunk.replace("\n\n", "\n").replace("\r", "")
        for page_num in range(1, len(pages)+1):
            cleaned = cleaned.replace(f"[PAGE {page_num}]", "")

        chunk_objs.append({
            "id": f"chunk_{i}",
            "text": cleaned.strip(),
            "start_page": start_page,
            "end_page": end_page
        })

    return chunk_objs
