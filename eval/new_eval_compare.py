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


def asof_merge_dfs(dfs, on, additional_merge_col):
    """
    Perform an asof merge on a list of dataframes followed by a regular merge on an additional column.
    """
    # Ensure all dataframes are sorted by the key
    for i, df in enumerate(dfs):
        dfs[i] = df.sort_values(by=on)

    # Start with the first dataframe
    merged_df = dfs[0]

    # Iteratively merge each dataframe
    for i, df in enumerate(dfs[1:]):
        # Drop overlapping columns except for the merge keys
        overlap_cols = [col for col in df.columns if col in merged_df.columns and col not in [on, additional_merge_col]]
        df_dropped = df.drop(columns=overlap_cols)

        # Ensure both DataFrames are sorted
        merged_df = merged_df.sort_values(by=on)
        df_dropped = df_dropped.sort_values(by=on)

        breakpoint()
        # Perform the asof merge
        merged_df = pd.merge_asof(merged_df, df_dropped, on=on, direction='nearest', suffixes=('', f'_df{i+1}'))

    breakpoint()
    # After all asof merges are done, perform a final regular merge on 'prompt'
    merged_df = pd.merge(merged_df, df, on=additional_merge_col, suffixes=('', f'_final'))

    return merged_df



def get_name(stem: str) -> str:
    return f"{stem}_sample"

def get_user_prompt_name(stem1: str, stem2: str) -> str:
    return f"{stem1}_vs_{stem2}_user_prompt"

def get_decision_name(stem1: str, stem2: str) -> str:
    return f"{stem1}_vs_{stem2}_decision"

def main(paths: List[str]):
    """
    Takes a list of paths that are all comparable (as in came from the same dataset and protocol).

    Creates a merged df with columns for the winners between each of the pairs. This should be suitable for learning curve generation downstream.
    """
    csv_paths = []
    existing_data = None
    for p in paths:
        p = Path(p)
        if p.suffix == '.csv':
            csv_paths.append(Path(p))
        elif p.suffix == '.pkl':
            with open(p, "rb") as f:
                existing_data = pickle.load(f)
        else:
            logging.warning(f"Unexpected file type {p.suffix} for path {p}")
    dfs = []
    for csvp in csv_paths:
        try:
            dfs.append(pd.read_csv(csvp))
        except Exception:
            logging.warning(f"Failed to read csv {csvp}")
    stems = [p.stem for p in csv_paths]
    existing_columns = [] if existing_data is None else existing_data.columns.tolist()
    dfs = [df.rename(columns={"sample": get_name(stem)}) for df, stem in zip(dfs, stems) if get_name(stem) not in existing_columns]
    # take the first 20 rows of each df for testing
    # dfs = [df.iloc[:20] for df in dfs]

    # merged_df = reduce(lambda left, right: pd.merge(left, right, on=["prompt", "step"], how="inner"), dfs)
    merged_df = asof_merge_dfs(dfs, on="step", additional_merge_col="prompt")
    if existing_data is not None:
        merged_df = pd.merge(merged_df, existing_data, on=["prompt", "step"], how="inner")
    all_stems = [colname[:-7] for colname in merged_df.columns.tolist() if colname.endswith("_sample")]
    print(f"{len(merged_df)=}")
    for stem_pair in tqdm(combinations(all_stems, 2)):
        print(f'Evaluating pair {stem_pair}')
        stem1, stem2 = stem_pair
        user_prompt_name = get_user_prompt_name(stem1, stem2)
        if user_prompt_name in merged_df.columns:
            logging.warning(f"Skipping pair {stem_pair} because it already exists")
            continue
        merged_df[user_prompt_name] = merged_df.apply(lambda row: get_user_prompt(row, get_name(stem1), get_name(stem2)), axis=1)
        breakpoint()
        completions = call_chats(zip(system_prompt_gen(), merged_df[user_prompt_name].tolist()), temperature=0.1)
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
    for stem in all_stems:
        pickle_name += f"{stem}-"
    pickle_name = pickle_name[:-1] + ".pkl"
    with open(pickle_name, "wb") as f:
        pickle.dump(merged_df, f)



if __name__ == '__main__':
    main(sys.argv[1:])
