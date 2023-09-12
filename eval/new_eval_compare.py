import sys
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
from functools import reduce
from itertools import combinations
import pickle
from typing import List
from viraj_fast_oai import call_chats

system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie. Output only that character and do not include any other characters or spaces."

user_prompt = "[User Question]\n{prompt}\n[The Start of Assistant A's Answer]\n{sample1}\n[The End of Assistant A's Answer]\n[The Start of Assistant B's Answer]\n{sample2}\n[The End of Assistant B's Answer]\n"

def system_prompt_gen():
    while True:
        yield system_prompt

def get_user_prompt(row, sample_1_name, sample_2_name):
    prompt = row["prompt"]
    sample1 = row[sample_1_name]
    sample2 = row[sample_2_name]
    return user_prompt.format(prompt=prompt, sample1=sample1, sample2=sample2)


def get_name(stem: str) -> str:
    return f"{stem}_sample"

def get_user_prompt_name(stem1: str, stem2: str) -> str:
    return f"{stem1}_vs_{stem2}_user_prompt"

def get_decision_name(stem1: str, stem2: str) -> str:
    return f"{stem1}_vs_{stem2}_decision"

def main(csv_paths: List[str]):
    """
    Takes a list of paths that are all comparable (as in came from the same dataset and protocol).

    Creates a merged df with columns for the winners between each of the pairs. This should be suitable for learning curve generation downstream.
    """
    csv_paths = [Path(p) for p in csv_paths]
    dfs = [pd.read_csv(p) for p in csv_paths]
    breakpoint()
    stems = [p.stem for p in csv_paths]
    dfs = [df.rename(columns={"sample": get_name(stem)}) for df, stem in zip(dfs, stems)]

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=["prompt", "step"], how="inner"), dfs)
    for stem_pair in tqdm(combinations(stems, 2)):
        print(f'Evaluating pair {stem_pair}')
        stem1, stem2 = stem_pair
        user_prompt_name = get_user_prompt_name(stem1, stem2)
        merged_df[user_prompt_name] = merged_df.apply(lambda row: get_user_prompt(row, get_name(stem1), get_name(stem2)), axis=1)
        completions = call_chats(zip(system_prompt_gen(), merged_df[user_prompt_name].tolist()))
        vals = []
        for i, dec in enumerate(completions):
            if dec == 'A':
                vals.append(stem1)
            elif dec == 'B':
                vals.append(stem2)
            elif dec == 'C':
                vals.append('Tie')
            else:
                logging.warning(f"Unexpected decision {dec} on row {i}")
                vals.append(dec)

        decision_name = get_decision_name(stem1, stem2)
        merged_df[decision_name] = vals
    pickle_name = ""
    for stem in stems:
        pickle_name += f"{stem}-"
    pickle_name = pickle_name[:-1] + ".pkl"
    with open(pickle_name, "wb") as f:
        pickle.dump(merged_df, f)



if __name__ == '__main__':
    main(sys.argv[1:])
