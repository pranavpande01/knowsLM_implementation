from pathlib import Path
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "dataset" / "delhi-food"
OUTPUT_JSON = PROJECT_ROOT / "chunks.json"


def discover_usable_pdfs(pdf_dir: Path, min_bytes: int = 80_000) -> list[str]:
    """Return absolute paths of PDFs likely useful for training/chunking."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    usable = [p for p in pdfs if p.stat().st_size >= min_bytes]

    if not usable:
        raise FileNotFoundError(f"No usable PDFs found in {pdf_dir}")

    print(f"Found {len(pdfs)} PDFs, using {len(usable)} (min_bytes={min_bytes}).")
    for p in usable:
        print(f"  - {p.name}")

    skipped = [p for p in pdfs if p not in usable]
    if skipped:
        print("Skipped small PDFs:")
        for p in skipped:
            print(f"  - {p.name} ({p.stat().st_size} bytes)")

    return [str(p) for p in usable]


def chunk_pdfs_for_qa(pdf_paths: list[str], chunk_size: int = 750, overlap: int = 100) -> list[Document]:
    all_chunks: list[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: list[Document], out_path: Path) -> None:
    chunk_data = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)


def load_chunks(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        chunk_data = json.load(f)
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in chunk_data]


if __name__ == "__main__":
    pdf_paths = discover_usable_pdfs(PDF_DIR)
    chunks = chunk_pdfs_for_qa(pdf_paths)
    save_chunks(chunks, OUTPUT_JSON)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_JSON}")

    # Optional reload sanity check
    reloaded = load_chunks(OUTPUT_JSON)
    print(f"Reloaded {len(reloaded)} chunks")
