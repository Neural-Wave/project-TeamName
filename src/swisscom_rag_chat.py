from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from swisscom_rag import SwisscomRAG
from langchain_core.output_parsers import StrOutputParser


class SwisscomRAGChat(SwisscomRAG):
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

        print('Sam:', self.invoke({'input':f'Express that you are starting anew with the conversation in {self.language}', 'context':None}, lang=False)[0])

    def update_summary(self, input_text: str, response: str):
        self.chat_history.append(f"""
          User: {input_text}
          Assistant: {response}
        """)

        history_text = "\n".join(self.chat_history)
        summary = self.summarization_chain.invoke({"chat_history": history_text})

        return summary

    def invoke(self, inputs: dict):
        # Define a function to get responses with memory
        question = inputs.get("input")
        # Inject the current summary as context
        context_with_summary = f"""
        Previous chat summary: {self.summary}
        
        New question: {question}
        """
        
        documents = self.retrieve_documents(question)

        response = self.chain.invoke({"input": context_with_summary, "context": documents})
        
        self.summary = self.update_summary(inputs, response)
        
        return response, documents

    
    def chat(self):
        """Run the chat interactive loop"""
        while True:
            # Get question from the user
            question = input('You: ')

            # Break the loop if the user wants to exit
            if question.lower() in ["exit", "sortie", "uscita"]:
                break

            # Use the chain to process the questio
            result, _ = self.invoke({"input": question})
            # wrapped_result = textwrap.fill(result, width=100)
            print("Sam:", result, "\n")
            
