from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path):
    """
    Extract text from a PDF, page by page.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    reader = PdfReader(str(pdf_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text.strip(),
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number
                }
            })

    if not documents:
        raise ValueError(
            "No extractable text found in the PDF. "
            "It may be scanned or contain no readable text."
        )

    return documents


def load_pdfs_from_directory(directory_path):
    """
    Extract text from every PDF in a directory.
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory.resolve()}"
        )

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {directory.resolve()}"
        )

    all_documents = []

    for pdf_file in pdf_files:
        documents = load_pdf(pdf_file)
        all_documents.extend(documents)

        print(
            f"Loaded {pdf_file.name}: "
            f"{len(documents)} pages with extractable text"
        )

    return all_documents