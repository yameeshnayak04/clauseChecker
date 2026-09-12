import os
import uuid
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import CORPUS_DIR, EMBEDDING_MODEL, COLLECTION_NAME

# PyMuPDF for PDF support
try:
    import pymupdf as fitz
except ImportError:
    fitz = None

def get_category_from_filename(filename: str) -> str:
    filename = filename.lower()
    if 'attendance' in filename:
        return 'attendance'
    if 'fee' in filename:
        return 'fees'
    if 'scholarship' in filename:
        return 'scholarship'
    if 'hostel' in filename:
        return 'hostel'
    return 'general'

def chunk_text_recursive(text: str, max_words: int = 300) -> list[dict]:
    words = text.split()
    if len(words) <= max_words:
        return [{"text": text, "start_word": 0, "end_word": len(words)}]
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        chunks.append({"text": " ".join(chunk_words), "start_word": i, "end_word": i + len(chunk_words)})
    return chunks

def process_md_file(filepath: Path) -> list[dict]:
    content = filepath.read_text(encoding='utf-8')
    sections = re.split(r'(^#{2,3} .+)', content, flags=re.MULTILINE)
    
    chunks = []
    current_section = "General"
    char_pos = 0
    
    for part in sections:
        if part.startswith('##'):
            current_section = part.strip('# ').strip()
            char_pos += len(part)
        else:
            text = part.strip()
            if not text:
                char_pos += len(part)
                continue
            
            words = text.split()
            if len(words) > 300:
                sub_chunks = chunk_text_recursive(text, 300)
                for sc in sub_chunks:
                    start_char = char_pos + text.find(sc['text'][:20]) if sc['text'] else char_pos
                    end_char = start_char + len(sc['text'])
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": sc['text'],
                        "source_file": filepath.name,
                        "category": get_category_from_filename(filepath.name),
                        "section_title": current_section,
                        "char_start": start_char,
                        "char_end": end_char,
                        "page_number": None,
                        "split_method": "recursive"
                    })
            else:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "source_file": filepath.name,
                    "category": get_category_from_filename(filepath.name),
                    "section_title": current_section,
                    "char_start": char_pos,
                    "char_end": char_pos + len(text),
                    "page_number": None,
                    "split_method": "structural"
                })
            char_pos += len(part)
            
    return chunks

def process_pdf_file(filepath: Path) -> list[dict]:
    if not fitz:
        print(f"Skipping PDF {filepath.name} because PyMuPDF is not installed.")
        return []
    
    doc = fitz.open(filepath)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if not text:
            continue
        
        words = text.split()
        if len(words) > 300:
            sub_chunks = chunk_text_recursive(text, 300)
            for sc in sub_chunks:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": sc['text'],
                    "source_file": filepath.name,
                    "category": get_category_from_filename(filepath.name),
                    "section_title": f"Page {page_num + 1}",
                    "char_start": 0,
                    "char_end": len(sc['text']),
                    "page_number": page_num + 1,
                    "split_method": "recursive"
                })
        else:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "source_file": filepath.name,
                "category": get_category_from_filename(filepath.name),
                "section_title": f"Page {page_num + 1}",
                "char_start": 0,
                "char_end": len(text),
                "page_number": page_num + 1,
                "split_method": "structural"
            })
            
    return chunks

def main():
    corpus_dir = Path(CORPUS_DIR)
    if not corpus_dir.exists():
        print(f"Corpus directory {corpus_dir} does not exist.")
        return

    all_chunks = []
    for filepath in sorted(corpus_dir.rglob('*')):
        if filepath.suffix == '.md':
            all_chunks.extend(process_md_file(filepath))
        elif filepath.suffix == '.pdf':
            all_chunks.extend(process_pdf_file(filepath))
            
    if not all_chunks:
        print("No chunks extracted.")
        return
    
    # Assign sequential, human-readable chunk IDs
    for i, chunk in enumerate(all_chunks):
        chunk['id'] = f"chunk_{i:04d}"
        
    print(f"Extracted {len(all_chunks)} chunks.")
    
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c['text'] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    from qdrant_client_factory import get_qdrant_client
    qdrant = get_qdrant_client()
    
    # Recreate collection
    try:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    print(f"Created collection '{COLLECTION_NAME}' with vector size {embeddings.shape[1]}.")
    
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(
            PointStruct(
                id=i,  # integer ID for Qdrant
                vector=embeddings[i].tolist(),
                payload=chunk
            )
        )
        
    # Upsert in batches
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        
    print("Ingestion complete.")
    
    # Summary
    file_counts = {}
    split_counts = {}
    for c in all_chunks:
        file_counts[c['source_file']] = file_counts.get(c['source_file'], 0) + 1
        split_counts[c['split_method']] = split_counts.get(c['split_method'], 0) + 1
        
    print("Chunks per file:", file_counts)
    print("Chunks per split method:", split_counts)

if __name__ == "__main__":
    main()

