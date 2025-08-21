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
import os
from dotenv import load_dotenv

import tkinter as tk
from tkinter import filedialog


load_dotenv("sec.env.example")  # Load environment variables from sec.env.example

# === 0. Authenticate ===
# Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN
login(HUGGINGFACE_TOKEN)

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Optional but clean


# === 1. Load and Split PDF ===
def load_and_split_pdf(pdf_path):
    poppler_path = r"D:\\practice\\Release-24.08.0-0\\poppler-24.08.0\\Library\\bin"
    print("🔍 Extracting text with PyPDFLoader...")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    print("🔍 Running OCR on empty pages (if any)...")
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    for i, page in enumerate(images):
        if i < len(raw_docs):
            if not raw_docs[i].page_content.strip():
                ocr_text = pytesseract.image_to_string(page)
                raw_docs[i] = Document(page_content=ocr_text, metadata={"source": pdf_path, "page": i + 1})
        else:
            ocr_text = pytesseract.image_to_string(page)
            raw_docs.append(Document(page_content=ocr_text, metadata={"source": pdf_path, "page": i + 1}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=30)
    return splitter.split_documents(raw_docs)


# === 2. Create Vector Store ===
def create_chroma_vectorstore(docs, persist_directory="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


# === 3. Load LLM (Gemini) ===
def load_llm(model_name="models/text-bison-001"):
    return GoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )


# === 4. Create QA Chain ===
def create_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")


# === 5. Chat Loop ===
def chat(qa_chain):
    print("✅ Ready! Ask your questions about the PDF. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        response = qa_chain.invoke({"query": query})["result"]
        print(f"Bot: {response}")



def request_pdf_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    return file_path

# === Main ===
if __name__ == "__main__":
    pdf_path = request_pdf_file()

    if not pdf_path:
        print("No file was selected. Exiting.")
    else:
        docs = load_and_split_pdf(pdf_path)
        vectordb = create_chroma_vectorstore(docs)
        llm = load_llm(model_name="gemini-1.5-flash")  # You can also try: "models/chat-bison-001"
        qa_chain = create_qa_chain(llm, vectordb)
        chat(qa_chain)

