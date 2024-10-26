import pandas as pd
from langchain_core.documents import Document
from pathlib import Path
import json
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(chunk_size=1000, chunk_overlap=100, path = "dataset/parsed_documents/"):
  """Helper gunction, load parsed documents and split them"""
  parsed_documents = Path(path).glob('*.json')
  parsed_documents_df = []

  for file in parsed_documents:
      parsed_documents_df.append(json.loads(file.read_text()))

  parsed_documents_df = pd.DataFrame(parsed_documents_df)
  print(parsed_documents_df.describe())

  documents = [
      Document(
          page_content=row['content'],
          metadata={'source': row['source'], 'language': row['language'], 'title': row['title']},
          id=str(uuid.uuid4())
          )
      for _, row in parsed_documents_df.iterrows()
  ]


  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
  )

  return text_splitter.split_documents(documents)