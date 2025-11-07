import streamlit as st
from pathlib import Path
import sys

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Research Assistant")
st.markdown("Ask questions about your scientific papers")

CHROMA_DIR = "chroma_db"
if not Path(CHROMA_DIR).exists():
    st.error("❌ ChromaDB not found. Please run indexing first:")
    st.code("python indexar_papers.py", language="bash")
    st.stop()

@st.cache_resource
def load_assistant():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3.1:8b"
    
    PROMPT_TEMPLATE = """You are a scientific research assistant. Use the context from the papers below to answer the question.

If the answer is not in the context, say "I couldn't find this information in the indexed papers."

Always cite which paper the information came from (file name).

Context from papers:
{context}

Question: {input}

Detailed answer:"""
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        
        llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.3
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        qa_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return qa_chain, vectorstore, retriever
        
    except Exception as e:
        st.error(f"Error loading assistant: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None
with st.spinner("🔄 Loading research assistant..."):
    result = load_assistant()
    if result and len(result) == 3:
        qa_chain, vectorstore, retriever = result
    else:
        qa_chain, vectorstore, retriever = None, None, None

if qa_chain is None:
    st.stop()

st.success("✅ Assistant ready!")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                st.markdown("---")
                st.markdown("**📄 Sources:**")
                for source in message["sources"]:
                    st.markdown(f"- {source}")
if prompt := st.chat_input("Ask your question about the papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer = qa_chain.invoke(prompt)
                
                source_docs = retriever.invoke(prompt)
                
                st.markdown(answer)
                
                unique_sources = set()
                for doc in source_docs:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        unique_sources.add(doc.metadata['source'])
                
                if unique_sources:
                    st.markdown("---")
                    st.markdown("**📄 Sources:**")
                    for source in sorted(unique_sources):
                        st.markdown(f"- {source}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": list(unique_sources)
                })
                
            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                st.error(error_msg)
                
                with st.expander("🐛 Show detailed error"):
                    import traceback
                    st.code(traceback.format_exc())
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
with st.sidebar:
    st.header("ℹ️ Information")
    
    try:
        num_docs = vectorstore._collection.count()
        st.metric("Indexed chunks", num_docs)
    except:
        st.metric("Indexed chunks", "N/A")
    
    st.markdown("---")
    
    st.markdown("**📚 How to use:**")
    st.markdown("1. Add PDFs to `papers/` folder")
    st.markdown("2. Run `python indexar_papers.py`")
    st.markdown("3. Ask questions here!")
    
    st.markdown("---")
    
    if st.button("🔄 Reindex papers", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("💡 Example questions"):
        st.markdown("""
        - What datasets are mentioned?
        - Compare SR3 and SRDiff approaches
        - What metrics were used?
        - Summarize super-resolution techniques
        - How to handle low-quality images?
        """)
    
    st.markdown("---")
    
    with st.expander("🔧 Debug info"):
        st.write(f"**ChromaDB path:** `{CHROMA_DIR}`")
        st.write(f"**Messages in history:** {len(st.session_state.messages)}")
        
        papers_path = Path("papers")
        if papers_path.exists():
            pdf_files = list(papers_path.glob("*.pdf"))
            st.write(f"**PDFs in papers/:** {len(pdf_files)}")
        else:
            st.write("**PDFs in papers/:** Folder not found")