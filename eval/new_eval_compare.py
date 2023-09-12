import sys
from pathlib import Path
import pandas as pd
import logging
import pickle
from viraj_fast_oai import call_chats

system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie. Output only that character and do not include any other characters or spaces."

user_prompt = "[User Question]\n{prompt}\n[The Start of Assistant A's Answer]\n{sample1}\n[The End of Assistant A's Answer]\n[The Start of Assistant B's Answer]\n{sample2}\n[The End of Assistant B's Answer]\n"

def get_user_prompt(row, sample_1_name, sample_2_name):
    prompt = row["prompt"]
    sample1 = row[sample_1_name]
    sample2 = row[sample_2_name]
    return user_prompt.format(prompt=prompt, sample1=sample1, sample2=sample2)


def main(csv1_path, csv2_path):
    csv1_path = Path(csv1_path)
    csv2_path = Path(csv2_path)
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df1.head()
    df2.head()
    print(f"{(df1['prompt'] == df2['prompt']).all()=}")
    stem1 = csv1_path.stem
    stem2 = csv2_path.stem
    name1 = f"{csv1_path.stem}_sample"
    name2 = f"{csv2_path.stem}_sample"
    df1 = df1.rename(columns={"sample": name1})
    df2 = df2.rename(columns={"sample": name2})

    merged_df = pd.merge(df1, df2, on=["prompt", "step"], how="inner")
    merged_df["User Prompt"] = merged_df.apply(lambda row: get_user_prompt(row, name1, name2), axis=1)
    user_prompt_list = merged_df["User Prompt"].tolist()
    system_prompt_gen = (system_prompt for _ in range(len(user_prompt_list)))
    print(f"Calling completions on {len(user_prompt_list)} prompts")
    completions = call_chats(zip(user_prompt_list, system_prompt_gen))
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
    merged_df["Decision"] = vals
    pickle_name = f"{csv1_path.stem}_vs_{csv2_path.stem}.pkl"
    with open(pickle_name, "wb") as f:
        pickle.dump(merged_df, f)



if __name__ == '__main__':
    main(*sys.argv[1:])
