from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split extracted PDF pages into smaller text chunks.

    Args:
        documents: List of dictionaries containing text and metadata.
        chunk_size: Maximum target size of each chunk, in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of dictionaries containing chunk text and metadata.
    """

    if not documents:
        raise ValueError("No documents provided for splitting.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(split_texts, start=1):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk": chunk_index
                }
            })

    return chunks