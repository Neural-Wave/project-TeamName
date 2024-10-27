import json
import pandas as pd
from tqdm.cli import tqdm
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


class Evaluator():
    def __init__(self, model, path="./data", name="test_data"):
        self.model = model
        self.path = path
        self.evaldata = self.load_inputs(name) # dataframe


    def load_inputs(self, name):
        print(f'Opening {self.path}/{name}.json')
        prompts = open(f'{self.path}/{name}.json')
        prompts_df = pd.DataFrame(json.load(prompts))
        return prompts_df


    def run_evaluations(self):
        results = []
        for input_ in tqdm(self.evaldata["input"]):
            result, _ = self.model.invoke({"input": input_})
            results.append(result)  

        self.evaldata["prediction"] = results


    def write_evaldata(self):
        name = f"./assets/botresults/results_{datetime.now().strftime('%H:%M')}.json"
        print(f"Writing {name}")
        self.evaldata.to_json(
            name,
            orient="records",
            indent=1
        )
        return name