from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from swisscom_rag import SwisscomRAG
from dotenv import load_dotenv

load_dotenv()

def main():
    vector_store = Chroma(
        collection_name="parsed_documents",  
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="chroma/swisscom_openai"
    )
  
    rag = SwisscomRAG(vector_store=vector_store)
        
    question = input("Ask something: ")
    
    results, documents = rag.invoke({"input": question})
    
    print(results)
    
    # for k, doc in enumerate(documents):
    #   print(f"k = {k};",doc.page_content)
    

if __name__ == "__main__":
    main()
