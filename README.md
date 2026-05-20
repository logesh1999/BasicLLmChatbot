# BasicLLmChatbot

A small Retrieval-Augmented Generation (RAG) demo that loads PDFs into Pinecone
and serves a Streamlit chat UI backed by a chat model (GROQ / ChatGroq).

**Requirements**
- See [requirements.txt](requirements.txt). Install with:

```bash
pip install -r requirements.txt
```

**Setup**
1. Create and activate a Python virtual environment (recommended).
2. Install dependencies (see above).
3. Configure environment variables — create `core/.env` or export the values:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
GROQ_API_KEY=your_groq_api_key
# Optional: OPENAI_API_KEY (if you plan to use OpenAI elsewhere)
```

**Ingest PDFs into Pinecone**
1. Place your PDF files in the `PDF/` folder at the project root.
2. Run the ingester to create the Pinecone index and upload embeddings:

```bash
python -m core.ingest
```

Notes:
- The ingester uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model.
- The code creates a serverless Pinecone index; ensure your Pinecone account supports this.

**Run the Streamlit app**

```bash
streamlit run app.py
```

Open the web UI, set `Top-k results` and `Temperature` in the sidebar, and ask questions
about your ingested PDFs.

**Implementation notes**
- Embeddings: `langchain-huggingface` + `sentence-transformers`.
- Vector DB: Pinecone via `pinecone-client` and `langchain-pinecone`.
- RAG LLM: `langchain-groq` / `ChatGroq` (requires `GROQ_API_KEY`).

If you want, I can pin package versions in `requirements.txt` for reproducible installs.