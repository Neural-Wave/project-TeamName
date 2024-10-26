from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lingua import Language, LanguageDetectorBuilder
from langchain_core.vectorstores import VectorStore

import textwrap

import re

PERSON = {
    'EN': 'You',
    'FR': 'Vous',
    'IT': 'Tu',
    'DE': 'Du'
}

class SwisscomRAGChat:
    def __init__(self, vector_store: VectorStore, language='EN', k=10):
        self.search_type = "bm25"  
        self.similarity_threshold = 0
        self.k = k
        self.temperature = 0.3
        self.language = language
        
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever()

        # Summarization parameters
        summarizer_llm = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
        summarization_prompt = PromptTemplate.from_template("""
        Given the chat history below, summarize it into a single, concise summary that retains context for the conversation:

        Chat history:
        {chat_history}

        Summary:
        """)
        
        self.summarization_chain = summarization_prompt | summarizer_llm | StrOutputParser()

        self.chat_history = []
        self.summary = ''
        
        # Initialize language model and prompt template
        self.main_llm = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_system_prompt()),
                ("human", "{input}"),
            ]
        )
        
        self.search_kwargs = {'k': self.k, 'filter': {'language': self.language}}
        self.chain = create_stuff_documents_chain(self.main_llm, prompt)

        print('Sam:', self.invoke({'input': f'Default interaction language will be {language}. '
                                            'Introduce yourself in this language. The default language may change if the user changes languages.'}, lang=False)[0])

    def detect_lang(self, question):
        languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.ITALIAN]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        detected_language = detector.detect_language_of(question).iso_code_639_1.name
        return detected_language
        

    def set_language(self, language):
        self.language = language

    def get_system_prompt(self):
        return """
You are a Swisscom website chatbot, designed to provide clear, professional assistance to potential and 
existing customers. Your primary goal is to offer thorough, helpful responses that address user needs 
while maintaining an engaging, friendly tone. Provide answers that are complete and informative, typically 
between 50-100 words, to offer a richer experience without overwhelming the user. Ensure responses are in natural, flowing sentences. 

Always include relevant links or suggest related Swisscom products when appropriate to the topic, even if not 
explicitly requested, to enhance the user’s experience. Ask follow-up questions to keep the conversation going, 
helping users explore Swisscom's offerings and find exactly what they need. For any unclear questions, kindly 
ask for clarification and, if necessary, offer to connect them to Swisscom customer support. Preferably append the URL at the very end on a new line.

If the user requests customer support, provide the following contact information: 0800 555 155, available 
Monday to Saturday from 8:00 to 20:00. Clearly explain your role is to assist with Swisscom-related questions 
and recommend support options as appropriate. Respond in the same language as the user, maintaining relevance, 
and avoid elaborating on unrelated topics.

{context}
"""

    def check_default_language(self, detected_language):
        if detected_language != self.language:
            if detected_language in ['EN', 'FR', 'IT', 'DE']:
                self.language = detected_language
                print('Changing default language to:', self.language)
                self.search_kwargs['filter']['language'] = self.language

    def invoke(self, inputs: dict, lang=True):
        question = inputs.get("input")
        
        detected_language = self.detect_lang(question)
        if lang:
            self.check_default_language(detected_language)
        
        # Retrieve documents based on the search type
        documents = self.retrieve_documents(question)
        
        result = self.chain.invoke({"input": question, "context": documents})
        # url = ''
        # try:
        # url = re.findall(r'(https://.+.html)"', documents[0].page_content)[0]
        # except:
        #     pass
        # result = f"{result} {url}"

        return result, documents

    def retrieve_documents(self, question):
        if self.search_type == "bm25":
            # BM25-based retrieval
            documents = self.retriever.get_relevant_documents(question)

        
        elif self.search_type == "similarity":
            # Similarity-based retrieval
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs=self.search_kwargs)
            documents = retriever.invoke(question)
        
        elif self.search_type == "rff":
            # Combine BM25 and similarity results with Reciprocal Rank Fusion
            similarity_retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs=self.search_kwargs)
            bm25_docs = self.retriever.get_relevant_documents(question)

            similarity_docs = similarity_retriever.invoke(question)

            # RFF: Combine BM25 and similarity results
            documents = self.reciprocal_rank_fusion(bm25_docs, similarity_docs)
        
        else:
            raise ValueError("Unsupported search type")
        
        return documents

    def reciprocal_rank_fusion(self, bm25_docs, similarity_docs):
        # RFF implementation: give higher rank to documents that appear in both lists
        combined_results = {}
        for rank, doc in enumerate(bm25_docs):
            combined_results[doc] = combined_results.get(doc, 0) + 1 / (rank + 1)
        for rank, doc in enumerate(similarity_docs):
            combined_results[doc] = combined_results.get(doc, 0) + 1 / (rank + 1)
        
        # Sort by combined RFF score
        sorted_documents = sorted(combined_results, key=combined_results.get, reverse=True)
        return sorted_documents[:self.k]

    def restart_chat_history(self):
        self.chat_history = []
        self.summary = ''
        print('Sam:', self.invoke({'input': f'Express that you are starting anew with the conversation in {self.language}'}, lang=False)[0])

    def invoke_with_memory(self, inputs, lang=True):
        question = inputs.get("input")
        context_with_summary = f"{self.summary}\n\n{inputs}"
        
        detected_language = self.detect_lang(question)
        if lang:
            self.check_default_language(detected_language)
        documents = self.retrieve_documents(question)
        response = self.chain.invoke({"input": question, "context": documents, "summary": context_with_summary})
        url = re.findall(r'(https://.+.html)"', documents[0].page_content)[0]
        response = f"{response} {url}"

        self.summary = self.update_summary(inputs, response)
        
        return response, documents

    def update_summary(self, input_text, response):
        self.chat_history.append(f"User: {input_text}\nAssistant: {response}")
        history_text = "\n".join(self.chat_history)
        new_summary = self.summarization_chain.invoke({"chat_history": history_text})
        return new_summary

    def chat(self):
        instructions = "Ask your question (or type 'exit' to quit): "
        if self.language != 'EN':
            instructions = self.invoke({'input': f'Translate {instructions} to {self.language}'}, lang=False)[0]

        while True:
            question = input(instructions)
            if question.lower() in ["exit", "sortie", "uscita"]:
                exit_msg = "Thanks for chatting today! Have a nice day."
                if self.language != 'EN':
                    exit_msg = self.invoke({'input': f"Translate {exit_msg} to {self.language}"}, lang=False)[0]
                print("Sam:", exit_msg)
                break

            try:
                print(f"{PERSON[self.language]}:", question, "\n")
                result, documents = self.invoke_with_memory({"input": question})
                wrapped_result = textwrap.fill(result, width=100)
                print("Sam:", wrapped_result, "\n")
            except Exception as e:
                print(f"Error: {e}")
