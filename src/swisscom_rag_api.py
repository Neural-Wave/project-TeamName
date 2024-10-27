from copy import deepcopy
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.output_parsers import StrOutputParser

from src.swisscom_rag import SwisscomRAG


class SwisscomRAGAPI(SwisscomRAG):
    def __init__(self, vector_store: VectorStore):
        super().__init__(vector_store)

        ##### Summarization LLM
        # Summarization parameters
        self.chat_history = []
        self.summary = ''

        summarizer_llm = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
        # Define the summarization prompt
        summarization_prompt = PromptTemplate.from_template("""
        Given the chat history below, summarize it into a single, concise summary that retains context for the conversation:

        Chat history:
        {chat_history}

        Summary:
        """)
        self.summarization_chain = summarization_prompt | summarizer_llm | StrOutputParser()

    def restart_chat_history(self):
        self.chat_history = []
        self.summary = ''

    def update_summary(self, input_text: str, response: str):
        self.chat_history.append(f"""
          User: {input_text}
          Assistant: {response}
        """)

        history_text = "\n".join(self.chat_history)
        summary = self.summarization_chain.invoke({"chat_history": history_text})

        return summary

    def invoke(self, messages: dict):

        if (len(messages['messages']) > 0):
            # Retrieve lasst message from queue
            last_message = messages['messages'][-1]
            assert last_message['role']=='user'
            question = last_message['content']

            # Inject the current summary as context
            question_with_summary = f"""
            Previous chat summary: {self.summary}
            
            New question: {question}
            """
            
            # Generate response
            documents = self.retrieve_documents(question)
            response = self.chain.invoke({"input": question_with_summary, "context": documents})
        else:
            # Generate greeting
            question = 'Introduce yourself in one sentence.'
            response = self.chain.invoke({"input": question, "context": []})

        
        self.summary = self.update_summary(question, response)

        updated_messages = deepcopy(messages)
        updated_messages['messages'].append({
            'role' : 'assistant',
            'content' : response
        })
        
        return updated_messages