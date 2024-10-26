import json
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect

from rag import SwisscomRAGChat

class HybridRetriever:
    def __init__(self, vector_store, documents_splitted, k: int = 3):
        self.vector_store = vector_store
        self.k = k
        self.documents = documents_splitted
        self.doc_texts = [doc.page_content for doc in documents_splitted]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.text_to_doc = {doc.page_content: doc for doc in documents_splitted}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def search(self, query: str, search_type: str = "hybrid", alpha: float = 0.5, filter_dict: Dict = None) -> List[Document]:
        """
        Enhanced search method with filtering support
        """
        # Apply language filter if provided
        if filter_dict and 'language' in filter_dict:
            filtered_docs = [
                doc for doc in self.documents 
                if doc.metadata.get('language') == filter_dict['language']
            ]
            if filtered_docs:
                self.documents = filtered_docs
                self.doc_texts = [doc.page_content for doc in filtered_docs]
                self.tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                self.text_to_doc = {doc.page_content: doc for doc in filtered_docs}

        if search_type == "hybrid":
            return self.hybrid_search(query, alpha)
        elif search_type == "dense":
            return self.vector_store.similarity_search(query, k=self.k, filter=filter_dict)
        elif search_type == "sparse":
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_k_idx = np.argsort(bm25_scores)[-self.k:][::-1]
            return [self.documents[i] for i in top_k_idx]
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def hybrid_search(self, query: str, alpha: float = 0.5) -> List[Document]:
        dense_results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=self.k * 2
        )
        dense_docs = {doc.page_content: score for doc, score in dense_results}
        
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_docs = {text: score for text, score in zip(self.doc_texts, bm25_scores)}
        
        combined_scores = defaultdict(float)
        dense_scores = self._normalize_scores(list(dense_docs.values()))
        bm25_scores_norm = self._normalize_scores(list(bm25_docs.values()))
        
        for doc_text, score in zip(dense_docs.keys(), dense_scores):
            combined_scores[doc_text] += alpha * score
        
        for doc_text, score in zip(bm25_docs.keys(), bm25_scores_norm):
            combined_scores[doc_text] += (1 - alpha) * score
        
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.k]
        
        return [self.text_to_doc[text] for text, _ in sorted_results]

class EnhancedRAGChain(SwisscomRAGChat):
    def __init__(self, vector_store, language='EN', k=10):
        super().__init__(vector_store, language, k)
        parsed_documents = Path('dataset/parsed_documents/').glob('*.json')
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


        self.hybrid_retriever = HybridRetriever(self.vector_store, documents_splitted, k)
        self.search_type = "hybrid"  # default to hybrid search
        self.alpha = 0.5  # default weight for hybrid search

    def set_search_parameters(self, search_type="hybrid", alpha=0.5):
        """
        Configure search parameters
        """
        self.search_type = search_type
        self.alpha = alpha
        print(f"Search type set to: {search_type}")
        if search_type == "hybrid":
            print(f"Hybrid alpha parameter set to: {alpha}")

    def invoke(self, inputs: dict):
        question = inputs.get("input")
        detected_language = detect(question).upper()
        
        # Use hybrid retriever instead of standard vector store retriever
        documents = self.hybrid_retriever.search(
            query=question,
            search_type=self.search_type,
            alpha=self.alpha,
            filter_dict={"language": detected_language}
        )
        
        result = self.chain.invoke({
            "input": question,
            "context": documents
        })
        
        return result, documents

    def invoke_with_memory(self, inputs):
        question = inputs.get("input")
        context_with_summary = f"{self.summary}\n\n{inputs}"
        detected_language = detect(question).upper()
        
        documents = self.hybrid_retriever.search(
            query=question,
            search_type=self.search_type,
            alpha=self.alpha,
            filter_dict={"language": detected_language}
        )
        
        response = self.chain.invoke({
            "input": question,
            "context": documents,
            "summary": context_with_summary
        })
        
        self.summary = self.update_summary(inputs, response)
        
        return response, documents

# Example usage
def main():
    # Load documents
    parsed_documents = Path('dataset/parsed_documents/').glob('*.json')
    parsed_documents_df = []

    for file in parsed_documents:
        parsed_documents_df.append(json.loads(file.read_text()))

    parsed_documents_df = pd.DataFrame(parsed_documents_df)

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="parsed_documents",
        embedding_function=embeddings,
        persist_directory="./chroma/swisscom_openai", 
    )

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

    # Add documents to vector store
    vector_store.add_documents(documents_splitted)

    # Initialize enhanced RAG chain
    rag_chain = EnhancedRAGChain(vector_store, documents_splitted, k=3)

    # Example queries
    queries = [
        "Tell me about blue mobile L prices",
        "What are the internet options for home?",
    ]

    # Test different search types
    search_types = ["dense", "sparse", "hybrid"]
    
    for search_type in search_types:
        print(f"\n=== Testing {search_type.upper()} search ===")
        rag_chain.set_search_parameters(search_type=search_type)
        
        for query in queries:
            print(f"\nQuery: {query}")
            response, documents = rag_chain.invoke({"input": query})
            print(f"Response: {response}")
            print("-" * 50)

if __name__ == "__main__":
    main()