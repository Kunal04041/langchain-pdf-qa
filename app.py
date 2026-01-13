import streamlit as st
from src.core.pdf_processor import PDFProcessor
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMService
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="InsightPDF Pro",
    page_icon="📄",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #38BDF8;
        color: #0F172A;
        border-radius: 8px;
        font-weight: 600;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #1E293B;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "processor" not in st.session_state:
    st.session_state.processor = PDFProcessor()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "llm" not in st.session_state:
    st.session_state.llm = LLMService()
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# --- UI LAYOUT ---
st.markdown('<h1 class="main-header">InsightPDF Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Document Intelligence & Analysis</p>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")
    st.divider()
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    
    if uploaded_file:
        if st.button("Upload PDF"):
            with st.spinner("Analyzing and indexing document..."):
                docs = st.session_state.processor.process_pdf(uploaded_file)
                st.session_state.vector_store.create_store(docs)
                st.session_state.indexed = True
                st.success(f"Indexed {len(docs)} chunks.")

    st.divider()
    if st.session_state.indexed:
        st.info("✅ Document ready for analysis")
    else:
        st.warning("⚠️ No document indexed")

# --- MAIN CONTENT ---
if not st.session_state.indexed:
    st.info("Please upload and index a PDF from the sidebar to begin your research.")
else:
    query = st.chat_input("Ask anything about the document...")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                # Search
                docs = st.session_state.vector_store.similarity_search(query)
                context = "\n---\n".join([d.page_content for d in docs])
                
                # Answer
                answer = st.session_state.llm.get_answer(context, query)
                st.markdown(answer)
                
                with st.expander("View Sources"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}**")
                        st.caption(doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})
