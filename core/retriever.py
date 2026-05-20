from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pathlib import Path
from typing import Any, Dict

# LangChain imports for RAG
try:
    from langchain_groq import ChatGroq
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
except Exception:
    # If langchain version differs, these imports may fail at runtime — keep module import safe.
    ChatGroq = None
    RetrievalQA = None
    PromptTemplate = None


def _load_env():
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
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set (env or core/.env)")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME not set (env or core/.env)")


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_retriever(k: int = 5):
    """Return a LangChain-style retriever backed by the Pinecone index.

    Args:
        k: number of top results to return.
    """
    # create Pinecone client and index reference
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # embeddings (must match embeddings used when ingesting)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # build vector store wrapper and return retriever
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever


if __name__ == "__main__":
    # simple interactive demo
    retriever = get_retriever(k=5)
    q = input("Query: ")
    try:
        # prefer public method
        docs = retriever._get_relevant_documents(q)
    except Exception:
        docs = retriever._get_relevant_documents(q)

    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        print(f"--- Result {i} (source={src}) ---")
        print(d.page_content[:1000])


def get_rag_chain(k: int = 5, temperature: float = 0.0) -> Any:
    """Create a Retrieval-Augmented Generation (RAG) chain using the Pinecone retriever.

    Returns a LangChain `RetrievalQA` chain configured with a Chat LLM and a prompt
    that instructs the model to cite sources and be concise.
    """
    if ChatGroq is None or RetrievalQA is None or PromptTemplate is None:
        raise RuntimeError("Required langchain chat/chain classes are not available in this environment")

    # ensure GROQ key present (ChatGroq reads from env)
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Set it in environment or core/.env to use the RAG chain")

    retriever = get_retriever(k=k)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert assistant. Use the provided context to answer the question. "
            "If the answer cannot be found in the context, say you don't know. "
            "Cite the source filenames from the context when relevant.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\n"
            "Answer concisely and include a short 'Sources:' section listing filenames."
        ),
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def rag_answer(chain: Any, query: str) -> Dict[str, Any]:
    """Run the RAG chain for `query` and return structured result with sources."""
    out = chain.run(query) if hasattr(chain, "run") else chain({"query": query})

    # normalize output — RetrievalQA.from_chain_type typically returns dict when return_source_documents=True
    if isinstance(out, dict):
        return out

    # otherwise try to construct a dict
    return {"result": out}
