import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PAPERS_DIR = "papers"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def index_papers():
    print("🔄 Starting indexing...")
    
    documents = []
    papers_path = Path(PAPERS_DIR)
    
    if not papers_path.exists():
        papers_path.mkdir()
        print(f"📁 Created folder '{PAPERS_DIR}'. Add your PDFs there!")
        return
    
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDFs found in '{PAPERS_DIR}'")
        return
    
    print(f"📚 Found {len(pdf_files)} papers")
    
    for pdf_path in pdf_files:
        print(f"  → Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source"] = pdf_path.name
            
            documents.extend(docs)
        except Exception as e:
            print(f"    ❌ Error processing {pdf_path.name}: {e}")
    
    print(f"✅ Total pages loaded: {len(documents)}")
    
    print("✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    print("🧮 Generating embeddings (may take a while)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'mps'}
    )
    
    if Path(CHROMA_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"✅ Index created in '{CHROMA_DIR}'")
    print("🎉 Indexing complete! Now you can ask questions.")

if __name__ == "__main__":
    index_papers()