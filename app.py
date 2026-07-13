"""
GigaCorp Customer Support RAG Agent
------------------------------------
A Streamlit chat app that answers customer questions using a local FAQ
knowledge base (RAG with FAISS), cites its sources, and remembers
conversation history across turns.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud (see README.md).
"""

import os
import streamlit as st

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="GigaCorp Support Assistant", page_icon="🛠️", layout="centered")
st.title("🛠️ GigaCorp Customer Support Assistant")
st.caption("Ask me about shipping, returns, business hours, or membership tiers. "
           "I remember our conversation and cite my sources.")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "gigacorp_faq.txt")

# --------------------------------------------------------------------------
# Sidebar: LLM provider + API key
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"], index=0)

    default_key = ""
    if provider == "OpenAI":
        default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    else:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

    api_key = st.text_input(
        f"{provider} API Key",
        value=default_key,
        type="password",
        help="Not stored anywhere. Falls back to Streamlit secrets if set."
    )

    if provider == "OpenAI":
        model_name = st.text_input("Model", value="gpt-4o-mini")
    else:
        model_name = st.text_input("Model", value="claude-3-5-haiku-20241022")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown("**Knowledge base:** `data/gigacorp_faq.txt`")
    with st.expander("Preview knowledge base"):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            st.text(f.read())

if not api_key:
    st.info("👈 Enter an API key in the sidebar to start chatting.")
    st.stop()

# --------------------------------------------------------------------------
# Build the vector store (cached across reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing knowledge base...")
def build_vectorstore(path: str):
    """Load the FAQ file, chunk it by section (preserving line numbers for
    citation), embed it, and build a FAISS vector store."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filename = os.path.basename(path)
    docs = []
    current_section = "General"
    section_start_line = 1
    buffer = []

    def flush(end_line):
        text = "".join(buffer).strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "section": current_section,
                        "start_line": section_start_line,
                        "end_line": end_line,
                    },
                )
            )

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("[Section:"):
            # flush previous section as a chunk before starting new one
            flush(i - 1)
            buffer = []
            current_section = stripped.strip("[]").replace("Section:", "").strip()
            section_start_line = i + 1
        else:
            buffer.append(line)
    flush(len(lines))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


vectorstore = build_vectorstore(DATA_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --------------------------------------------------------------------------
# Build the LLM
# --------------------------------------------------------------------------
def get_llm():
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0.2)

llm = get_llm()

# --------------------------------------------------------------------------
# History-aware retriever: rewrites follow-up questions using chat history
# e.g. "How much does it cost?" -> "How much does shipping to India cost?"
# --------------------------------------------------------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might "
               "reference context in the chat history, formulate a standalone "
               "question which can be understood without the chat history. "
               "Do NOT answer the question, just reformulate it if needed, "
               "otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful, concise customer support assistant for a company called GigaCorp. "
     "Answer the user's question using ONLY the following retrieved context from the "
     "GigaCorp FAQ knowledge base. If the answer is not in the context, say you don't "
     "have that information and suggest contacting support@gigacorp-example.com. "
     "Be friendly and direct.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --------------------------------------------------------------------------
# Conversation memory (persists per browser session)
# --------------------------------------------------------------------------
msgs = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --------------------------------------------------------------------------
# Chat UI
# --------------------------------------------------------------------------
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi! I'm the GigaCorp support assistant. Ask me anything about "
                         "shipping, returns, business hours, or membership tiers.")

for msg in msgs.messages:
    role = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(msg.content)
        # Re-display citations stored alongside AI messages, if any
        if role == "assistant" and msg.additional_kwargs.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg.additional_kwargs["sources"]:
                    st.markdown(f"- **{s['source']}** — *{s['section']}* "
                                f"(lines {s['start_line']}-{s['end_line']})")

if user_input := st.chat_input("Ask a question, e.g. 'Do you ship to India?'"):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "streamlit_session"}},
                )
                answer = result["answer"]
                source_docs = result.get("context", [])

                st.markdown(answer)

                sources = []
                if source_docs:
                    with st.expander("📚 Sources"):
                        seen = set()
                        for d in source_docs:
                            key = (d.metadata["section"], d.metadata["start_line"], d.metadata["end_line"])
                            if key in seen:
                                continue
                            seen.add(key)
                            sources.append({
                                "source": d.metadata["source"],
                                "section": d.metadata["section"],
                                "start_line": d.metadata["start_line"],
                                "end_line": d.metadata["end_line"],
                            })
                            st.markdown(f"- **{d.metadata['source']}** — *{d.metadata['section']}* "
                                        f"(lines {d.metadata['start_line']}-{d.metadata['end_line']})")

                # Attach sources to the last AI message so they persist on rerender
                if msgs.messages:
                    msgs.messages[-1].additional_kwargs["sources"] = sources

            except Exception as e:
                st.error(f"Something went wrong: {e}")
