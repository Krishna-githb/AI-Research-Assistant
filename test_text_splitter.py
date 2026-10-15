from src.document_loader import load_pdfs_from_directory
from src.text_splitter import split_documents


def main():
    print("===== TEXT CHUNKING TEST =====\n")

    # Step 1: Extract text from research papers
    documents = load_pdfs_from_directory("data/documents")

    print(f"\nExtracted pages: {len(documents)}")

    # Step 2: Split extracted text into chunks
    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Display a sample chunk
    print("\n===== FIRST CHUNK =====\n")

    first_chunk = chunks[0]

    print("Source:", first_chunk["metadata"]["source"])
    print("Page:", first_chunk["metadata"]["page"])
    print("Chunk number:", first_chunk["metadata"]["chunk"])
    print("Chunk length:", len(first_chunk["text"]))

    print("\nChunk text preview:\n")
    print(first_chunk["text"][:500])

    # Step 4: Verify chunk sizes
    oversized_chunks = [
        chunk for chunk in chunks
        if len(chunk["text"]) > 1000
    ]

    print("\n===== VALIDATION =====")
    print("Chunks exceeding 1000 characters:", len(oversized_chunks))

    if oversized_chunks:
        raise AssertionError("Some chunks exceed the configured size.")

    print("\n===== TEXT CHUNKING SUCCESSFUL =====")


if __name__ == "__main__":
    main()