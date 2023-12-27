import pandas as pd
import json
import sys
import re

def extract_placeholder(data):
    # Regular expression pattern to match both formats
    pattern = r'(?i)^(?:This joke is about )?(.*?)(?:\.\s*)?$'

    # Searching for the pattern in the given data
    match = re.match(pattern, data)
    if match:
        return match.group(1)  # Returns the captured group
    else:
        return None  # No match found


def main(src, dst):
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
    data.to_csv(dst, index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
