# GigaCorp Customer Support RAG Agent

A Streamlit chat app that answers customer questions about a fictional
company ("GigaCorp") using Retrieval-Augmented Generation (RAG) over a local
FAQ knowledge base. It cites its sources (section + line numbers) and
remembers conversation context across turns.

## Features
- **Backend/orchestration:** Python + LangChain (history-aware retrieval chain)
- **LLM:** OpenAI or Anthropic (switchable in the sidebar)
- **Knowledge base:** `data/gigacorp_faq.txt` — mock FAQ covering shipping,
  returns, business hours, and service tiers — embedded with
  `sentence-transformers` and indexed in a local **FAISS** vector store
- **Citations:** every answer shows the FAQ section and exact line range it
  came from
- **Memory:** follow-up questions (e.g. "how much does it cost?" after
  asking about shipping to India) are automatically understood in context
- **UI:** Streamlit chat interface

## 1. Run locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your OpenAI or Anthropic key

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). You can also
just paste your API key directly into the sidebar instead of using secrets.toml.

## 2. Deploy free on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (must include `app.py`,
   `requirements.txt`, and the `data/` folder).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → select your repo, branch (`main`), and set
   **Main file path** to `app.py`.
4. Click **"Advanced settings"** → **Secrets**, and paste:
   ```toml
   OPENAI_API_KEY = "sk-...your-key..."
   ANTHROPIC_API_KEY = "sk-ant-...your-key..."
   ```
   (You only strictly need the one for the provider you plan to use.)
5. Click **Deploy**. First build takes a few minutes (downloading the
   embedding model). You'll get a public URL like
   `https://your-app-name.streamlit.app`.

### Alternative: Hugging Face Spaces (also free)
1. Create a new Space → SDK: **Streamlit**.
2. Upload `app.py`, `requirements.txt`, and `data/gigacorp_faq.txt`.
3. In Space **Settings → Repository secrets**, add `OPENAI_API_KEY` or
   `ANTHROPIC_API_KEY`.
4. The Space builds and serves automatically.

## 3. How it satisfies the assignment requirements

| Requirement | Where |
|---|---|
| Python + LangChain orchestration | `app.py` — `create_history_aware_retriever`, `create_retrieval_chain` |
| Reliable LLM API | OpenAI (`langchain-openai`) or Anthropic (`langchain-anthropic`), switchable in sidebar |
| Mock FAQ knowledge base | `data/gigacorp_faq.txt` — shipping, returns, hours, tiers |
| Local vector store | FAISS (`faiss-cpu`), built once and cached |
| Chat UI | Streamlit `st.chat_message` / `st.chat_input` |
| Sources & citations | Each answer's "📚 Sources" expander shows section name + exact line numbers |
| Conversational memory | `StreamlitChatMessageHistory` + `RunnableWithMessageHistory`; follow-ups are rewritten using chat history before retrieval |
| Free hosting | Streamlit Community Cloud or Hugging Face Spaces (see above) |

## Project structure
```
rag_project/
├── app.py                          # Main Streamlit app
├── requirements.txt
├── data/
│   └── gigacorp_faq.txt            # Mock knowledge base
├── .streamlit/
│   └── secrets.toml.example        # Template — copy to secrets.toml
└── README.md
```

## Notes
- The embedding model (`all-MiniLM-L6-v2`) runs locally/free — no API key
  needed for embeddings, only for the chat LLM.
- To swap in your own knowledge base, replace `data/gigacorp_faq.txt` with
  your own text using the same `[Section: Name]` heading format so citations
  keep working, or point `DATA_PATH` in `app.py` at a PDF loader if you'd
  rather use a PDF (LangChain's `PyPDFLoader` works as a drop-in).
