from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lingua import Language, LanguageDetectorBuilder
from langchain_core.vectorstores import VectorStore
from copy import deepcopy

PERSON = {
  'EN' : 'You',
  'FR' : 'Vous',
  'IT' : 'Tu',
  'DE' : 'Du'
}

import textwrap


class SwisscomRAGChat:
    def __init__(self, vector_store: VectorStore, language='EN', k = 10):
        self.search_type = "similarity" # "similarity" - "similarity-threshold" - "mmr"
        self.k = k
        self.temperature = 0.3
        self.language = language
        
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever()

        # Summarization parameters
        summarizer_llm = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
        # Define the summarization prompt
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
        
        # Define the prompt with system and human messages
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_system_prompt()),
                ("human", "{input}"),
            ]
        )
        
        self.search_kwargs = {'k': self.k, 'filter':{'language': self.language}}
        # Initialize the chain using the llm and prompt
        self.chain = create_stuff_documents_chain(self.main_llm, prompt)

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
        existing customers.  Your responses should be limited to information provided in the context and avoid lists. 
        Include relevant links only when specifically mentioned in the context, and try to continue the conversation by 
        asking relevant follow-up questions. If unsure of an answer, ask if the user would like to be forwarded to 
        Swisscom customer support and clarify that your role is to assist only with Swisscom-related inquiries. 
        If the user requests customer support, provide contact information: 0800 555 155, available Monday to Saturday 
        from 8:00 to 20:00. Subtly suggest other relevant Swisscom products or services when possible. Respond in the 
        same language as the user and, for irrelevant inquiries, avoid sharing additional information.

        {context}
        """


    def check_default_language(self, detected_language):
        # Detect the language of the question
        if detected_language != self.language:
          if detected_language in ['EN', 'FR', 'IT', 'DE']:
            self.language = detected_language
            print('Chaning default language to:', self.language)

            self.search_kwargs['filter']['language'] = self.language

    def invoke(self, inputs: dict, lang=True):
        question = inputs.get("input")
        
        detected_language = self.detect_lang(question)

        if lang:
          self.check_default_language(detected_language)
        
        # Do a search on the vector store
        retriever = self.vector_store.as_retriever(
          search_type=self.search_type, 
          search_kwargs=self.search_kwargs
        )
        
        # Get relevant documents
        documents = retriever.invoke(question)
           
        # Pass the filtered documents to the question-answer chain
        response = self.chain.invoke({"input": question, "context": documents})
        
        return response, documents
      

    def restart_chat_history(self):
        self.chat_history = []
        self.summary = ''

        print('Sam:', self.invoke({'input':f'Express that you are starting anew with the conversation in {self.language}', 'context':None}, lang=False)[0])


    def invoke_with_memory(self, inputs, lang=True):
        # Define a function to get responses with memory
        
        question = inputs.get("input")
        # Inject the current summary as context
        context_with_summary = f"{self.summary}\n\n{inputs}"
        
        # Detect the language of the question
        detected_language = self.detect_lang(question)
        if lang:
          self.check_default_language(detected_language)

        # Do a search on the vector store
        retriever = self.vector_store.as_retriever(
          search_type=self.search_type, 
          search_kwargs=self.search_kwargs
        )

        # Retrieve relevant documents and generate answer
        documents = retriever.invoke(question)
        
        response = self.chain.invoke({"input": question, "context": documents, "summary": context_with_summary})
        
        # Update memory summary after each interaction
        self.summary = self.update_summary(inputs, response)
        
        return response, documents

      
    def invoke_batch(self, messages):

        last_message = messages['messages'][-1]
        assert last_message['role']=='user', 'User is not last message'
        question = last_message['content']
        print(question)
        
        # Detect the language of the question
        detected_language = self.detect_lang(question)
        self.check_default_language(detected_language)

        # Do a search on the vector store
        retriever = self.vector_store.as_retriever(
          search_type=self.search_type, 
          search_kwargs=self.search_kwargs
        )

        # Retrieve relevant documents and generate answer
        documents = retriever.invoke(question)
        response = self.chain.invoke({"input": question, "context": documents})
        
        # Update messages
        updated_messages = deepcopy(messages)
        response_parsed = {
          'user' : 'assistant',
          'content' : response
        }
        updated_messages['messages'].append(response_parsed)
        return updated_messages


    def update_summary(self, input_text, response):
        # Append new user and assistant message to the history
        self.chat_history.append(f"User: {input_text}\nAssistant: {response}")

        # Join the history and summarize
        history_text = "\n".join(self.chat_history)
        new_summary = self.summarization_chain.invoke({"chat_history": history_text})

        return new_summary


    def chat(self):
      instructions = "Ask your question (or type 'exit' to quit): "
      # info_msg = "You can find out more in"
      if self.language != 'EN':
        instructions = self.invoke({'input':f'Translate {instructions} to {self.language}', 'context':None}, lang=False)[0]
        # info_msg = self.invoke({'input':f'Translate {info_msg} to {self.language}', 'context':None}, lang=False)[0]

      print('Sam:', self.invoke({'input':f'Default interaction language will be {self.language}. \
          Introduce yourself in this language. The default language may change if the user changes languages.', 'context':None}, lang=False)[0])

      while True:
        # Get question from the user
        question = input(instructions)

        # Break the loop if the user wants to exit
        if question.lower() in ["exit", "sortie", "uscita"]:
            exit_msg = "Thanks for chatting today! Have a nice day."
            if self.language != 'EN':
              exit_msg = self.invoke({'input':f"Translate {exit_msg} to {self.language}", 'context':None}, lang=False)[0]
            print("Sam:",  exit_msg)
            break

        # Use the chain to process the question
        try:
            print(f"{PERSON[self.language]}:", question, "\n")
            result, documents = self.invoke_with_memory({"input": question})
            wrapped_result = textwrap.fill(result, width=100)
            print("Sam:", wrapped_result, "\n")
            # if len(documents):
            #   print(f"{info_msg} {documents[0].metadata['source']} \n")
        except Exception as e:
            print(f"Error: {e}")
