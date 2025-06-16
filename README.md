# Complete Retrieval-Augmented Generation (RAG) AI System with LangChain & Hugging Face
In this demo, we build an End-to-End Retrieval-Augmented Generation (RAG) AI System using LangChain for RAG framework, FAISS for vector store, Hugging Face model as LLM and Gradio for frontend UI. RAG is a framework that improves LLMs by retrieving relevant documents before generating an answer. It typically ingest documents, retrieve similar documents to the ones queried, and the retrieved documents are passed to an LLM to generate an answer.



## Requirements
1. Hugging Face Transformers:   Local LLM for generation
2. LangChain:                   Framework for RAG pipeline
3. FAISS:                       Vector Store
4. PyTorch:                     Python package for Deep Learning and GPU-accelerated tensor computations
5. pymupdf:                  PDF Support
6. sentence-transformers:       For word embeddings
7. Gradio:                      For Web UI



## Set up Environment

### Install system packages (if needed) & Create Virtual Environment
Some Python packages are still catching up with full compatibility for Python3.12 (latest). You may have to use a previous version of Python like 3.10.

#### With Latest Python version 3.12
```sh
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
python3 -m venv venv
source venv/bin/activate
```

#### With Python version 3.10
```sh
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip -y
python3.10 -m venv rag_env
source rag_env/bin/activate
```

### Install the Required Packages
```sh
pip install -r requirements.txt 
```
Note: If langchain gives you issues, install or upgrade the langchain cli with command
```sh
pip install langchain-cli
pip install -U langchain-huggingface
langchain upgrade
```


## Set up config.py to Load the LLM Model
Add script to the config.py file
You don't have GPU, use llm model like: "google/flan-t5-base" (lightweight and CPU-friendly)
 
```sh
# config.py
from transformers import pipeline

#device=0 is for GPU, device=-1 is for CPU
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256, device=0)
```

## Build the PDF Ingest Script
This script reads PDFs from uploaded_pdfs directory, chunks them, embeds them, and stores them in a FAISS vectorstore. 

```sh
# ingest.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import fitz 
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def read_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    doc.close()
    return text

def ingest_pdfs(directory="uploaded_pdfs"):
    all_docs = []

    if not os.path.exists(directory):
        print("❌ No 'uploaded_pdfs/' directory found.")
        return

    pdfs = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    if not pdfs:
        print("❌ No PDF files found in directory.")
        return

    for filename in pdfs:
        file_path = os.path.join(directory, filename)
        print(f"📄 Reading: {filename}")
        text = read_pdf_text(file_path).strip()

        if not text:
            print(f"⚠️ Skipping empty PDF: {filename}")
            continue

        #chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text(text)
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " "]
            )
        chunks = splitter.split_text(text)
        if not chunks:
            print(f"⚠️ No chunks created from: {filename}")
            continue

        docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
        all_docs.extend(docs)

    if not all_docs:
        print("❌ No documents to embed. Check PDF content.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(all_docs, embeddings)
    vectordb.save_local("vectorstore")
    print("✅ Vectorstore saved successfully.")

# ✅ Make sure the script actually runs
if __name__ == "__main__":
    ingest_pdfs()
```


## Sample Test PDF 
Add a sample pdf file (any article or tech doc) in the uploaded_pdfs directory or create a sample pdf with the script below

```sh
# sample.py

import os
from fpdf import FPDF

# Ensure directory exists
os.makedirs("uploaded_pdfs", exist_ok=True)

def txt_to_pdf(txt_path, output_path="uploaded_pdfs/sample.pdf"):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.readlines()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in content:
        pdf.cell(200, 10, txt=line.strip(), ln=True)

    pdf.output(output_path)
    print(f"✅ PDF saved to {output_path}")

# Example usage
if __name__ == "__main__":
    txt_file_path = "sample.txt" 
    txt_to_pdf(txt_file_path)
```

### Run the Script
```sh
python sample.py
```


### Run the Document Ingestion Script
```sh
python ingest.py
```


## Build the Gradio App
This script provides a Web UI to upload PDFs and interact with the RAG system

```sh
# app.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import shutil
import gradio as gr

from config import load_llm_pipeline
from ingest import ingest_pdfs
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Prepare upload directory
UPLOAD_DIR = "uploaded_pdfs"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Vector Store
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        VECTORSTORE_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )


# Load RAG pipeline
def load_rag_chain():
    vectordb = load_vectorstore()
    pipe = load_llm_pipeline()
    llm = HuggingFacePipeline(pipeline=pipe)
    return RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectordb.as_retriever(search_kwargs={"k": 1}), 
        chain_type="stuff", 
        return_source_documents=True
    )

# Build knowledge base
def handle_pdfs(files):
    if files:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for file in files:
            # file is a NamedString or tempfile path
            file_path = os.path.join(UPLOAD_DIR, os.path.basename(str(file)))
            shutil.copyfile(file, file_path)  # ✅ safely copy the uploaded file
        ingest_pdfs()
        return "✅ Knowledge base created."
    return "⚠️ No files uploaded."


# RAG Q&A
def query_pdf_rag(question):
    rag_chain = load_rag_chain()
    result = rag_chain.invoke({"query": question})

    # Print what the retriever gave us (for debug)
    print("🔎 Retrieved Documents:")
    for doc in result["source_documents"]:
        print(f"\n📄 {doc.metadata.get('source')}:\n{doc.page_content[:300]}\n---")

    answer = result["result"]

    # Trim long source content in Gradio display
    sources = "\n\n".join(
        f"**{doc.metadata.get('source', 'Unknown')}**\n{doc.page_content[:500]}..."
        for doc in result["source_documents"]
    )
    return answer, sources

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG AI with Hugging Face & PDF Upload")

    with gr.Row():
        file_upload = gr.File(label="Upload PDF files", file_types=[".pdf"], file_count="multiple")
        upload_button = gr.Button("Build Knowledge Base")
        upload_status = gr.Textbox(label="", interactive=False)

    upload_button.click(fn=handle_pdfs, inputs=[file_upload], outputs=[upload_status])

    query = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="🤖 Answer", lines=3, interactive=False)
    sources = gr.Textbox(label="📚 Sources", lines=10, interactive=False)

    query.submit(fn=query_pdf_rag, inputs=[query], outputs=[answer, sources])

# Launch
if __name__ == "__main__":
    demo.launch()
```

### Run the Gradio App
```sh
python app.py
```
On your browser:
```sh
localhost:7860
```

On Gradio UI, 
1. Upload pdf file
2. Click "Build Knowledge Base" and 
3. Ask a question e.g. 
        What is LangChain?
        What are some features of LangChain?
        How does LangChain build apps?
Press Enter


