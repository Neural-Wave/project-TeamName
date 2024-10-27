from enum import Enum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import json

from dotenv import load_dotenv

load_dotenv()

from src.swisscom_rag_api import SwisscomRAGAPI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = Chroma(
  collection_name="parsed_documents",
  embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
  persist_directory="../../chroma/swisscom_openai"
)

rag = SwisscomRAGAPI(vector_store=vector_store)

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

@app.post("/v1/chat/completions")
def completions(request: CompletionsRequest):
    
    messages = []

    for message in request.messages:
      messages.append({"role":message.role,"content":message.content})
    
    results = rag.invoke({"messages":messages})
    
    return results