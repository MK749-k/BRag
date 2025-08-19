import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document

from huggingface_hub import login
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os

# === Set API Keys ===
HUGGINGFACE_TOKEN = "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"
GOOGLE_API_KEY = "AIzaSyDTzrCQwzk61mJyuq8WNOwFuhKbJhpm73Y"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
login(HUGGINGFACE_TOKEN)

poppler_path = r"D:\\practice\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin"  # Change to your actual path


# === PDF Loader with OCR ===
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    for i, page in enumerate(images):
        if i < len(raw_docs):
            if not raw_docs[i].page_content.strip():
                ocr_text = pytesseract.image_to_string(page)
                raw_docs[i] = Document(page_content=ocr_text, metadata={"page": i + 1})
        else:
            ocr_text = pytesseract.image_to_string(page)
            raw_docs.append(Document(page_content=ocr_text, metadata={"page": i + 1}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(raw_docs)


# === Vector DB and Chain Setup ===
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=docs, embedding=embeddings)

def load_llm():
    return GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

def create_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# === Streamlit App ===
st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        docs = load_and_split_pdf(tmp_pdf_path)
        vectordb = create_vectorstore(docs)
        llm = load_llm()
        qa_chain = create_qa_chain(llm, vectordb)

    st.success("✅ PDF Processed! Ask your questions below:")

    user_query = st.text_input("Ask a question:")
    if user_query:
        response = qa_chain.invoke({"query": user_query})["result"]
        st.write("🤖", response)
