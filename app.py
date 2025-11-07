import gradio as gr
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from template import PROMPT_TEMPLATE

# Configuration
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"

# Load assistant once at startup
print("🚀 Loading research assistant...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Create RAG chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Assistant ready!")
    
except Exception as e:
    print(f"❌ Error loading assistant: {e}")
    import traceback
    traceback.print_exc()
    rag_chain = None
    retriever = None

def ask_question(question, history):
    """Process question and return answer"""
    if not rag_chain:
        return "❌ Assistant not loaded. Run: python indexar_papers.py"
    
    if not question.strip():
        return "Please ask a question."
    
    try:
        # Get answer
        answer = rag_chain.invoke(question)
        
        # Get sources separately
        docs = retriever.invoke(question)
        
        # Extract unique sources
        unique_sources = set()
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                unique_sources.add(doc.metadata['source'])
        
        # Format response with sources
        if unique_sources:
            answer += "\n\n📄 **Sources:**\n"
            for source in sorted(unique_sources):
                answer += f"- {source}\n"
        
        return answer
        
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

# Create Gradio interface
with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Research Assistant")
    gr.Markdown("Ask questions about your scientific papers")
    
    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        avatar_images=(None, "🤖")
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask your question about the papers...",
            show_label=False,
            scale=4
        )
        submit = gr.Button("Send", scale=1, variant="primary")
    
    with gr.Accordion("ℹ️ Information", open=False):
        gr.Markdown("""
        **How to use:**
        1. Add PDFs to `papers/` folder
        2. Run `python indexar_papers.py`
        3. Ask questions here!
        
        **Example questions:**
        - What datasets are mentioned?
        - Compare SR3 and SRDiff approaches
        - What metrics were used?
        - Summarize super-resolution techniques
        """)
    
    clear = gr.Button("🗑️ Clear Chat")
    
    def user_message(user_msg, history):
        return "", history + [[user_msg, None]]
    
    def bot_response(history):
        user_msg = history[-1][0]
        bot_msg = ask_question(user_msg, history)
        history[-1][1] = bot_msg
        return history
    
    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )