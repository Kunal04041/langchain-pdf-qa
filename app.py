import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("PDF Document QA")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Read PDF content
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create documents for vectorstore
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create embeddings and vectorstore dynamically
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Set up the LLM and QA chain
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=generator)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input("Ask a question about your PDF:")

    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
        st.write("**Answer:**")
        st.write(answer)

else:
    st.info("Please upload a PDF file to start.")
