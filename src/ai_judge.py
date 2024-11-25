from pydantic import BaseModel
from typing import Literal
from tqdm.cli import tqdm
import pandas as pd
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

class Decision(BaseModel):
    result: Literal['A','B','Tie']
    short_reason: str

class AIJudge():
    def __init__(self, input_file, input_base, output_file):

        self.input_file = pd.read_json(input_file)
        self.input_base = pd.read_json(input_base)        
        self.output_file = output_file
          
        self.client = OpenAI()

    def evaluate_responses(self, question, response_a, response_b) -> Decision:
        messages = [
            {"role": "system", "content": """
              You are an LLM judge evaluating two chatbot responses to the same question. Please follow these criteria:
              1. **Language consistency**: Does the chatbot answer in the language of the user? Does it adapt if the user asks to switch languages?
              2. **Answer consistency**: Does the response accurately follow from any prior conversation context? Does it handle unrelated questions appropriately?
              3. **Relevance**: Is the answer inherently related to the question?
              4. **Factuality**: Is the response based on the correct and relevant information?
              5. **Correlation**: Does the answer mention relevant references, if applicable?

              For each question, choose the better response. Announce the result as A, B or Tie.
              """},
              {"role": "user", "content": f"""
                Question: "{question}"

                Response A: {response_a}
                Response B: {response_b}

                Please decide which response better meets the criteria above.
              """}
        ]

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=Decision,
        )

        return completion.choices[0].message.parsed

    def save_filename(self, s):
        return "".join(x for x in s if x.isalnum())

    def judge(self):

        # Main evaluation loop
        results = []

        their_records = self.input_base.to_dict(orient='records')
        our_records = self.input_file.to_dict(orient='records')

        for _, (entry_a, entry_b) in tqdm(enumerate(zip(their_records, our_records))):
            question = entry_a['input']
            response_a = entry_a['prediction']
            response_b = entry_b['prediction']
            
            # Get evaluation from the LLM
            decision = self.evaluate_responses(question, response_a, response_b)

            if decision.result == 'A':
                score = -1
            elif decision.result == 'B':
                score = +1
            elif decision.result == 'Tie':
                score = 0

            # Append result
            results.append({
                "question": question,
                "base": response_a,
                "ours": response_b,
                "reason": decision.short_reason,
                "score": score,
            })
            
            # Output results every time
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=1)

        print("Evaluation complete. Results saved to ", self.output_file)

    def compute_overall_score(self):
        results = pd.read_json(self.output_file)
        return results.score.mean()

