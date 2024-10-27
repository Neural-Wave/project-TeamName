from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings

from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store_chroma = Chroma(
        collection_name="parsed_documents",
        embedding_function=embeddings,
        persist_directory="chroma/swisscom_riccardo", 
    )

    directory_loader = DirectoryLoader(
        path="dataset/parsed_documents",  # Directory with JSON files
        glob="*.json",  # Only load files with .json extension
        loader_cls=JSONLoader,  # Specify JSONLoader as the loader class
        loader_kwargs={
            "jq_schema": "{page_content: .content, metadata: {source: .source, title: .title, language: .language}}",
            "text_content": False,
        },
        show_progress=True
    )

    documents = directory_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200, add_start_index=True
    )

    documents_splitted = text_splitter.split_documents(documents)

    batch_size = 50  

    for i in tqdm(range(0, len(documents_splitted), batch_size)):
        batch = documents_splitted[i:i + batch_size]
        vector_store_chroma.add_documents(batch)
        
if __name__ == "__main__":
    main()