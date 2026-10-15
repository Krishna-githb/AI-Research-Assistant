from src.document_loader import load_pdfs_from_directory

PDF_DIRECTORY = "data/documents"


def main():
    print("===== PDF LOADER TEST =====\n")

    documents = load_pdfs_from_directory(PDF_DIRECTORY)

    print(f"\nTotal pages extracted: {len(documents)}")

    print("\n===== FIRST PAGE EXTRACTED =====\n")

    first_document = documents[0]

    print("Source:", first_document["metadata"]["source"])
    print("Page:", first_document["metadata"]["page"])

    print("\nExtracted text preview:\n")
    print(first_document["text"][:1000])

    print("\n===== PDF EXTRACTION SUCCESSFUL =====")


if __name__ == "__main__":
    main()