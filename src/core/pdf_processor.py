from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process_pdf(self, file_path_or_stream):
        """
        Extracts text from PDF and splits into manageable chunks.
        """
        reader = PdfReader(file_path_or_stream)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        chunks = self.splitter.split_text(text)
        # Convert to Document objects for LangChain compatibility
        return [Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks]
