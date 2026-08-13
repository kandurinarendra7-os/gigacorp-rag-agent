import os
import json
import streamlit as st

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="GigaCorp Support Assistant", page_icon="🛠️", layout="centered")

# Custom CSS injection for high-impact UI styling improvements
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------
# Sidebar: LLM provider, API key, and Professional Tools
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("LLM Provider", ["Groq", "OpenAI", "Anthropic"], index=0)

    default_key = ""
    if provider == "Groq":
        default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    elif provider == "OpenAI":
        default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    else:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

    api_key = st.text_input(
        f"{provider} API Key",
        value=default_key,
        type="password",
        help="Not stored anywhere. Falls back to Streamlit secrets if set."
    )

    if provider == "Groq":
        model_name = st.text_input("Model", value="llama-3.3-70b-versatile")
    elif provider == "OpenAI":
        model_name = st.text_input("Model", value="gpt-4o-mini")
    else:
        model_name = st.text_input("Model", value="claude-3-5-haiku-20241022")

    st.divider()
    st.header("📊 Professional Tools")
    
    # Session export feature
    msgs_export_chk = StreamlitChatMessageHistory(key="chat_messages")
    if len(msgs_export_chk.messages) > 0:
        chat_data = []
        for m in msgs_export_chk.messages:
            chat_data.append({"role": m.type, "content": m.content})
        
        json_str = json.dumps(chat_data, indent=2)
        st.download_button(
            label="📥 Export Chat (JSON)",
            data=json_str,
            file_name="gigacorp_chat_history.json",
            mime="application/json"
        )

        md_str = "# GigaCorp Support Chat Transcript\n\n"
        for m in msgs_export_chk.messages:
            role_title = "User" if m.type == "human" else "Assistant"
            md_str += f"**{role_title}:**\n{m.content}\n\n---\n\n"
        st.download_button(
            label="📥 Export Chat (Markdown)",
            data=md_str,
            file_name="gigacorp_chat_history.md",
            mime="text/markdown"
        )
    else:
        st.caption("Export options will appear once a conversation starts.")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown("**Knowledge base:** `data/gigacorp_faq.txt`")
    if os.path.exists(DATA_PATH := os.path.join(os.path.dirname(__file__), "data", "gigacorp_faq.txt")):
        with st.expander("Preview knowledge base"):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                st.text(f.read())

st.title("🛠️ GigaCorp Customer Support Assistant")
st.caption("Ask me about shipping, returns, business hours, or membership tiers. "
           "I remember our conversation, cite my sources, and support audio playback & feedback.")

if not api_key:
    st.info(f"👈 Enter a {provider} API key in the sidebar to start chatting.")
    st.stop()

# --------------------------------------------------------------------------
# Build the vector store (cached across reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing knowledge base...")
def build_vectorstore(path: str):
    """Load the FAQ file, chunk it by section, embed it, and build a FAISS vector store."""
    if not os.path.exists(path):
        st.error(f"Data file not found at {path}")
        st.stop()
        
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
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.2)
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0.2)

try:
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to initialize LLM client: {e}")
    st.stop()

# --------------------------------------------------------------------------
# History-aware retriever
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
# Conversation memory
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
# Chat UI with Working Audio Component via Streamlit Components HTML
# --------------------------------------------------------------------------
import streamlit.components.v1 as components

def render_tts_audio_player(text: str, unique_key: str):
    clean_text = json.dumps(text)
    html_code = f"""
    <div style="display: flex; gap: 10px; align-items: center; margin-top: 8px; font-family: sans-serif;">
        <button id="playbtn_{unique_key}" style="background-color: #2b5c8f; color: white; border: none; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 14px; font-weight: 500;">🔊 Read Aloud</button>
        <button id="stopbtn_{unique_key}" style="background-color: #f0f2f6; color: #31333F; border: 1px solid #d6d9dc; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 14px; font-weight: 500;">⏹️ Stop</button>
        <span id="status_{unique_key}" style="font-size: 12px; color: #666;"></span>
    </div>
    <script>
    const text_{unique_key} = {clean_text};
    const playBtn_{unique_key} = document.getElementById('playbtn_{unique_key}');
    const stopBtn_{unique_key} = document.getElementById('stopbtn_{unique_key}');
    const status_{unique_key} = document.getElementById('status_{unique_key}');

    playBtn_{unique_key}.onclick = function() {{
        if (!('speechSynthesis' in window)) {{
            status_{unique_key}.innerText = 'Speech synthesis not supported.';
            return;
        }}
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text_{unique_key});
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.activeUtterance = utterance;
        
        utterance.onstart = function() {{ status_{unique_key}.innerText = 'Playing...'; }};
        utterance.onend = function() {{ status_{unique_key}.innerText = ''; }};
        utterance.onerror = function() {{ status_{unique_key}.innerText = 'Error playing speech.'; }};

        window.speechSynthesis.speak(utterance);
    }};

    stopBtn_{unique_key}.onclick = function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            status_{unique_key}.innerText = '';
        }}
    }};
    </script>
    """
    components.html(html_code, height=50)

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi! I'm the GigaCorp support assistant. Ask me anything about "
                         "shipping, returns, business hours, or membership tiers.")

for idx, msg in enumerate(msgs.messages):
    role = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role):
        st.markdown(msg.content)
        
        if role == "assistant":
            if msg.additional_kwargs.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg.additional_kwargs["sources"]:
                        st.markdown(f"- **{s['source']}** — *{s['section']}* "
                                    f"(lines {s['start_line']}-{s['end_line']})")
            
            render_tts_audio_player(msg.content, f"msg_{idx}")

            cols = st.columns([1, 1, 10])
            with cols[0]:
                if st.button("👍", key=f"thumb_up_{idx}"):
                    st.toast("Thank you for your positive feedback!", icon="✅")
            with cols[1]:
                if st.button("👎", key=f"thumb_down_{idx}"):
                    st.toast("Thank you for your feedback. We will improve!", icon="⚠️")

if user_input := st.chat_input("Ask a question, e.g. 'Do you ship to India?'"):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("GigaCorp assistant is thinking and searching knowledge base..."):
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
                                "start_line": d.metadata[
                                    "start_line"
                                ],
                                "end_line": d.metadata["end_line"],
                            })
                            st.markdown(f"- **{d.metadata['source']}** — *{d.metadata['section']}* "
                                        f"(lines {d.metadata['start_line']}-{d.metadata['end_line']})")

                if msgs.messages:
                    msgs.messages[-1].additional_kwargs["sources"] = sources
                    
                render_tts_audio_player(answer, f"msg_{len(msgs.messages)-1}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
