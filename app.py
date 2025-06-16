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