from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from swisscom_rag_chat import SwisscomRAGChat
from dotenv import load_dotenv

load_dotenv()

def main():
    vector_store = Chroma(
        collection_name="parsed_documents",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="chroma/swisscom_openai"
    )
  
    rag = SwisscomRAGChat(vector_store=vector_store)
    
    rag.chat()

if __name__ == "__main__":
    main()
