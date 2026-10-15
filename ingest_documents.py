from src.document_loader import load_pdfs_from_directory
from src.text_splitter import split_documents
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager


PDF_DIRECTORY = "data/documents"


def main():
    print("=" * 50)
    print("AI RESEARCH ASSISTANT - DOCUMENT INGESTION")
    print("=" * 50)

    # Step 1: Extract text from PDFs
    print("\n[1/4] Extracting PDF text...")

    documents = load_pdfs_from_directory(PDF_DIRECTORY)

    print(f"Total pages extracted: {len(documents)}")

    # Step 2: Split text into chunks
    print("\n[2/4] Splitting documents into chunks...")

    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")

    embedding_manager = EmbeddingManager()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)

    print(f"Total embeddings generated: {len(embeddings)}")

    # Step 4: Store chunks and embeddings
    print("\n[4/4] Storing chunks in ChromaDB...")

    vector_store = VectorStoreManager()

    stored_count = vector_store.add_documents(
        chunks,
        embeddings
    )

    print(f"Chunks processed: {stored_count}")
    print(f"Total chunks in database: {vector_store.count_documents()}")

    print("\n" + "=" * 50)
    print("DOCUMENT INGESTION SUCCESSFUL")
    print("=" * 50)


if __name__ == "__main__":
    main()