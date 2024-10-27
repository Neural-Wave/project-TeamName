from enum import Enum
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

from swisscom_rag import SwisscomRAG
from pydantic import BaseModel

app = FastAPI()

vector_store = Chroma(
  collection_name="parsed_documents",
  embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
  persist_directory="chroma/swisscom_openai"
)

rag = SwisscomRAG(vector_store=vector_store)

class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class Message(BaseModel):
  role: Role
  content: str

class CompletionsRequest(BaseModel):
    messages: list[Message]

class CompletionsResponse(BaseModel):
    messages: list[Message]

@app.post("/v1/chat/completions", response_model=CompletionsResponse)
def completions(request: CompletionsRequest):
    results, documents = rag.invoke_batch({
      "messages":[{"role":"user","content": "Hello Swisscom"}]
    })
    return {
      "messages": results,
      "documents": documents,
    }