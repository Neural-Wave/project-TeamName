import json
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm.cli import tqdm

from dotenv import load_dotenv
load_dotenv()

from swisscom_rag import SwisscomRAG

INPUT_FILE = "data/input.json"
OUTPUT_FILE = "data/output.json"

def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vector_store = Chroma(
      collection_name="parsed_documents",
      embedding_function=embeddings,
      persist_directory="chroma/swisscom_openai"
    )
    
    rag = SwisscomRAG(vector_store=vector_store)
    
    f = open(INPUT_FILE)

    input_data = json.load(f)

    input_data_df = pd.DataFrame(input_data)

    predictions = []

    for input in tqdm(input_data_df["input"]):
      prediction, _ = rag.invoke({"input": input})
      predictions.append(prediction)
      
    input_data_df["prediction"] = predictions

    input_data_df.to_json(
        OUTPUT_FILE,
        orient="records",
        indent=1
    )
  
if __name__ == "__main__":
    main()