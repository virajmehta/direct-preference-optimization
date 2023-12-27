import pandas as pd
from tqdm import tqdm
import json
from copy import deepcopy


df = pd.read_csv('jokes.csv')
json_fields = dict(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": "You are an assistant tasked with extracting what a joke is about. Assume we will put your response into the template \"This joke is about %s\", so respond only with the value for %s. DO NOT INCLUDE \"This joke is about\""},
                ])

with open('jokes.jsonl', 'w') as f:
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        json_row = deepcopy(json_fields)
        json_row['messages'].append({'role': 'user', 'content': row['Joke']})
        json_str = json.dumps(json_row)
        f.write(json_str)
        f.write('\n')
