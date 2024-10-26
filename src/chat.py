from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rag import SwisscomRAGChat
from dotenv import load_dotenv
import os

load_dotenv()

def main():
  
    
    vector_store = Chroma(
        collection_name="parsed_documents",  # Same collection name as the existing store
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="chroma/swisscom_riccardo"  # Path to the existing store's directory
    )
  
    # Initialize the SwisscomRAGChat with the specified persistence directory
    rag = SwisscomRAGChat(vector_store=vector_store)
    
    # Start the chat interface
    rag.chat()

if __name__ == "__main__":
    main()
