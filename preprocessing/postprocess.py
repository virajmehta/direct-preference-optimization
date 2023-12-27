import pandas as pd
import json
import sys
import re
from sklearn.model_selection import train_test_split

def extract_placeholder(data):
    pattern = r'(?i)^(?:This joke is about )?(.*?)(?:\.\s*)?$'
    match = re.match(pattern, data)
    if match:
        return match.group(1)
    else:
        return None

def main(src, train_dst, test_dst):
    prompts = []
    responses = []
    with open(src, 'r') as f:
        for line in f:
            elt = json.loads(line)
            joke = elt[0]['messages'][1]['content']
            about = elt[1]['choices'][0]['message']['content']
            about = extract_placeholder(about)
            prompt = f"Tell me a joke about {about}."
            prompts.append(prompt)
            responses.append(joke)

    data = pd.DataFrame({'prompt': prompts, 'response': responses})

    # Split the data into a training set and a test set (90/10 split)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Write the datasets to separate CSV files
    train_data.to_csv(train_dst, index=False)
    test_data.to_csv(test_dst, index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

