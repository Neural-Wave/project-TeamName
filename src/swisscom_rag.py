from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever

SYSTEM_PROMPT = """
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

class SwisscomRAG:
    def __init__(self, vector_store: VectorStore):
        self.temperature = 0.3
        self.vector_store = vector_store
        self.main_llm = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
        self.system_prompt = SYSTEM_PROMPT

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        
        self.chain = create_stuff_documents_chain(self.main_llm, prompt)

    def retrieve_documents(self, question):
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 30})

        documents = retriever.invoke(question)
        
        if(len(documents)):
          retriever_bm25 = BM25Retriever.from_documents(documents)
          retriever_bm25.k = 3
          documents = retriever_bm25.invoke(question)

        return documents
      
    def invoke(self, inputs: dict):
        question = inputs.get("input")
        
        documents = self.retrieve_documents(question)
        
        result = self.chain.invoke({"input": question, "context": documents})

        return result, documents