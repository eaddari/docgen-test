from pathlib import Path
import json
import chromadb
from chromadb.config import Settings

def load_embeddings(embeddings_output, chunks_output):
    ids = []
    embeddings = []
    metadatas = []
    embeddings_root = Path(embeddings_output)
    chunks_root = Path(chunks_output)
    for embedding_file in embeddings_root.rglob('*_embeddings.json'):
        rel_path = embedding_file.relative_to(embeddings_root)
        print(f"Processing embedding file: {rel_path}")
        with open(embedding_file, 'r') as f:
            vectors = json.load(f)
        chunk_file = chunks_root / rel_path
        chunk_file = chunk_file.with_name(chunk_file.name.replace('_embeddings.json', '_chunks.txt'))
        print(f"Looking for chunk file: {chunk_file} | Exists: {chunk_file.exists()}")
        if chunk_file.exists():
            with open(chunk_file, 'r', encoding='utf-8') as chunk_f:
                chunk_texts = [line.strip() for line in chunk_f]
            print(f"  Vectors: {len(vectors)}, Chunks: {len(chunk_texts)}")
        else:
            chunk_texts = []
            print("  Chunk file not found!")
        for idx, vector in enumerate(vectors):
            chunk_text = chunk_texts[idx] if idx < len(chunk_texts) else ""
            ids.append(f"{rel_path}_{idx}")
            embeddings.append(vector)
            metadatas.append(
                {"file_path": str(rel_path),
                 "chunk_index": idx,
                 "text": chunk_text}
            )
    return ids, embeddings, metadatas

if __name__ == "__main__":


    embeddings_output = "embeddings_output"
    chunks_output = "chunks_output"
    client = chromadb.PersistentClient(path="C:\\Users\\FJ138WZ\\OneDrive - EY\\Documents\\test DocGen\\chroma")

    collection = client.get_or_create_collection(name="doc_embeddings")

    ids, embeddings, metadatas = load_embeddings(embeddings_output, chunks_output)

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"Added {len(ids)} embeddings to the collection.")

