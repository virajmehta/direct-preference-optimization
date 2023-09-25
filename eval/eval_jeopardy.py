import sys
from pathlib import Path
import pandas as pd
import logging
import pickle
from viraj_fast_oai import call_chats
from tqdm import tqdm

system_prompt = "You are a judge on whether a contestant answer to Jeopardy is correct given a correct answer. If you don't see the correct answer it is not correct. Answer 'Yes' or 'No' is sufficient. Please don't use any other words."
good_csv_fn = "further_sft_1.csv"

def get_user_prompt(row):
    return f"Correct answer is: \"{row['correct_answer']}\". Contestant Answer: \"{row['sample_only']}\""

def main(csv_dir_path):

    csv_dir_path = Path(csv_dir_path)
    good_csv_path = csv_dir_path / good_csv_fn
    good_df = pd.read_csv(good_csv_path)

    for csv_path in tqdm(csv_dir_path.iterdir()):
        print(f'processing {csv_path}')
        if not str(csv_path).endswith('.csv'):
            continue
        df = pd.read_csv(csv_path)
        missing_df = df[pd.isna(df['correct_answer'])]
        missing_df.loc[:, 'correct_answer'] = good_df['correct_answer'][missing_df.index]
        df['sample_only'] = df.apply(lambda row: row['sample'][len(row['prompt']):], axis=1)

        df['user_prompt'] = df.apply(get_user_prompt, axis=1)
        user_prompt_list = df['user_prompt'].tolist()
        system_prompt_gen = (system_prompt for _ in range(len(user_prompt_list)))
        completions = call_chats(zip(system_prompt_gen, user_prompt_list))
        df['correct'] = completions
        # df.to_csv('test.csv', index=False)
        df.to_csv(str(csv_path), index=False)

if __name__ == '__main__':
    main(sys.argv[1])
