import json
import pandas as pd
from tqdm.cli import tqdm
from datetime import datetime


class Evaluator():
    def __init__(self, model):
        self.model = model
        self.path = "./assets/evaldata"
        self.evaldata = self.load_evaldata(name="all") # dataframe


    def load_evaldata(self, name):
        print(f'Opening {self.path}/{name}.json')
        f = open(f'{self.path}/{name}.json')
        evaldata = json.load(f)
        evaldata_df = pd.DataFrame(evaldata)
        print(evaldata_df.head())
        return evaldata_df


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