from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except Exception:
    env_path = Path(__file__).parent / ".env"

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    k, v = line.split("=", 1)

                    os.environ.setdefault(
                        k.strip(),
                        v.strip().strip('"').strip("'")
                    )

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

if not pinecone_api_key:
    raise RuntimeError("PINECONE_API_KEY not set")

if not pinecone_index_name:
    raise RuntimeError("PINECONE_INDEX_NAME not set")


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384


def create_pinecone_index():
    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if pinecone_index_name in existing_indexes:
        print(f"Deleting existing index: {pinecone_index_name}")

        pc.delete_index(pinecone_index_name)

        print("Index deleted successfully")

    print(f"Creating new index: {pinecone_index_name}")

    pc.create_index(
        name=pinecone_index_name,
        dimension=VECTOR_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Waiting for index to become ready...")

    while True:
        description = pc.describe_index(pinecone_index_name)

        if description.status["ready"]:
            break

    print("Pinecone index is ready")

    return pc.Index(pinecone_index_name)


def ingest_pdf_to_pinecone(pdf_folder_path):
    documents = []

    pdf_files = list(Path(pdf_folder_path).glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_folder_path}")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    index = create_pinecone_index()

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    print("Uploading vectors to Pinecone...")

    vectorstore.add_documents(chunks)

    print("Upload completed")

    return len(chunks)


if __name__ == "__main__":
    pdf_folder_path = Path(__file__).parent / "PDF"

    num_chunks = ingest_pdf_to_pinecone(pdf_folder_path)

    print(
        f"\nSuccessfully ingested {num_chunks} chunks "
        f"into Pinecone index '{pinecone_index_name}'"
    )
