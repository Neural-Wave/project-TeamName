from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from pandas as pd
import json
import uuid
from rank_bm25 import BM25Okapi
import re

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseVectorStore(ABC):
    @abstractmethod
    def get_retriever(self, **kwargs) -> BaseRetriever:
        """Return a retriever instance"""
        pass

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str, embeddings, persist_directory: str):
        self.vector_store_ = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    def get_retriever(self, **kwargs) -> BaseRetriever:
        return self.vector_store_.as_retriever(
            search_type=kwargs.get('search_type', 'similarity'),
            search_kwargs=kwargs.get('search_kwargs', {})
        )

class BM25VectorStore(BaseVectorStore):
    def __init__(self, collection_name: str):
        
        parsed_documents = Path(f'dataset/{collection_name}/').glob('*.json')
        parsed_documents_df = []

        for file in parsed_documents:
            parsed_documents_df.append(json.loads(file.read_text()))

        parsed_documents_df = pd.DataFrame(parsed_documents_df)

        # Convert DataFrame rows to Documents
        documents = [
            Document(
                page_content=row['content'],
                metadata={'source': row['source'], 'language': row['language'], 'title': row['title']},
                id=str(uuid.uuid4())
            )
            for _, row in parsed_documents_df.iterrows()
        ]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2024, chunk_overlap=200, add_start_index=True
        )
        documents_splitted = text_splitter.split_documents(documents)

        self.doc_texts = [doc.page_content for doc in documents_splitted]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.text_to_doc = {doc.page_content: doc for doc in documents_splitted}


    def get_retriever(self, **kwargs) -> BaseRetriever:
        retriever = self.vector_store
        if 'k' in kwargs.get('search_kwargs', {}):
            retriever.k = kwargs['search_kwargs']['k']
        return retriever

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())