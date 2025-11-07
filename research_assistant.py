from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"

PROMPT_TEMPLATE = """You are a scientific research assistant. Use the context from the papers below to answer the question.

If the answer is not in the context, say "I could not find this information in the indexed papers."

Always cite which paper the information came from (file name).

Papers context:
{context}

Question: {input}

Detailed answer:"""

class ResearchAssistant:
    def __init__(self):
        print("🚀 Loading research assistant...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'mps'}
        )
        
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )
        
        self.llm = Ollama(
            model=LLM_MODEL,
            temperature=0.3
        )
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.retriever = retriever
        
        print("✅ Assistant ready!\n")
    
    def ask(self, question):
        print(f"🤔 Question: {question}\n")
        print("🔍 Searching papers...")
        
        docs = self.retriever.invoke(question)
        
        answer = self.qa_chain.invoke(question)
        
        print(f"\n💡 Answer:\n{answer}\n")
        
        print("📄 Sources:")
        unique_sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in unique_sources:
                print(f"  • {source}")
                unique_sources.add(source)
        
        print("\n" + "="*80 + "\n")
        
        return answer

def interactive_mode():
    assistant = ResearchAssistant()
    
    print("💬 Interactive mode started!")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['sair', 'exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            assistant.ask(question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    interactive_mode()