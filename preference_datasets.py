import datasets
import time
import pandas as pd
import time
import torch
import json
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import asyncio
import openai
import nltk
from nltk.corpus import cmudict
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


load_dotenv()
syllable_dict = cmudict.dict()



def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.

       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.

       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.

       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data

def get_jeopardy(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    print(f'Loading Jeopardy! dataset from file...')
    with open(f'data/{split}_jeopardy_data.json', 'r') as f:
        data = json.load(f)
    '''
    data is of the form

    {'category': 'HISTORY', 'air_date': '2004-12-31', 'question': "'For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory'", 'value': '$200', 'answer': 'Copernicus', 'round': 'Jeopardy!', 'show_number': '4680', 'wrong_answer': 'Kepler'}
    '''
    # TODO: will need to iterate on prompts to some extent
    def make_prompt_and_responses(elt):
        category = elt['category']
        question = elt['question']
        value = elt['value']
        answer = elt['answer']
        wrong_answer = elt['wrong_answer']
        prompt = f'{category}, for {value}: {question}'
        # change null token to empty string
        # responses = [answer, 'null', wrong_answer]
        responses = [answer, "", wrong_answer]
        pairs = [(0, 1), (0, 2), (1, 2)]
        # take a single sample
        pairs = [random.choice(pairs)]
        return prompt, dict(responses=responses, pairs=pairs, sft_target=answer)
    all_data = {}
    for row in tqdm.tqdm(data, desc="Processing Jeopardy!", disable=silent):
        prompt, data = make_prompt_and_responses(row)
        all_data[prompt] = data
    return all_data


def get_jokes(split: str, silent: bool=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    print(f'Loading Jokes dataset from file...')
    df = pd.read_csv(f'data/joke_data_{split}.csv')
    all_data = {}
    for idx, row in tqdm.tqdm(df.iterrows(), desc="Processing Jokes", disable=silent, total=df.shape[0]):
        if 'r/jokes' in row['prompt'] or 'r/jokes' in row['response']:
            continue
        # prompt = "Instruct: " + row['prompt'] + "\nOutput: "
        prompt = row['prompt'] + "\n"
        responses = [row['response']]
        sft_target = row['response']
        pairs = []
        all_data[prompt] = dict(responses=responses, pairs=pairs, sft_target=sft_target)
    return all_data


def get_haikus(split: str, silent: bool=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    print(f'Loading Haiku dataset from file...')
    df = pd.read_csv(f'data/haiku_{split}.csv')
    all_data = {}
    for idx, row in tqdm.tqdm(df.iterrows(), desc="Processing Jokes", disable=silent, total=df.shape[0]):
        # prompt = "Instruct: " + row['prompt'] + "\nOutput: "
        prompt = f'Write a haiku containing the words "{row["keywords"]}".\n'
        haiku = row['text']
        responses = [haiku]
        sft_target = haiku
        pairs = []
        all_data[prompt] = dict(responses=responses, pairs=pairs, sft_target=sft_target)
    return all_data


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == 'jeopardy':
        data = get_jeopardy(split, silent=silent, cache_dir=cache_dir)
    elif name == 'jokes':
        data = get_jokes(split, silent=silent, cache_dir=cache_dir)
    elif name == 'haikus':
        data = get_haikus(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        # TODO: make this sensible for active iteration
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Optional[Dict]:
    """Tokenize a single batch element.

       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    if tokenizer.eos_token_id in prompt_tokens['input_ids']:
        print(f"Prompt contains EOS token: {prompt}")
        return None
    if tokenizer.eos_token_id in chosen_tokens['input_ids']:
        print(f"Chosen response contains EOS token: {chosen}")
        return None
    if tokenizer.eos_token_id in rejected_tokens['input_ids']:
        print(f"Rejected response contains EOS token: {rejected}")
        return None

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch



def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def call_api(client, model, messages):
    response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.)
    return response

async def get_winner(client, model, system_message, prompt, a, a_prime):
    user_message = f"Instruction: {prompt}, A: {a}, B: {a_prime}"
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    try:
        response = await call_api(client, model, messages)
    except tenacity.RetryError as e:
        print(e)
        return None
    # response = await call_api(model, messages)
    choice = response.choices[0].message.content
    return choice == "A"

async def get_winners(dataset_name: str, prompts: List[str], actions: List[str], a_primes: List[str], model='gpt-4-1106-preview') -> List[bool]:
    assert dataset_name in ('jokes', 'haikus')
    if dataset_name == 'jokes':
        return await get_winners_jokes(prompts, actions, a_primes, model=model)
    elif dataset_name == 'haikus':
        return await get_winners_haikus(prompts, actions, a_primes, model=model)

async def get_winners_jokes(prompts: List[str], actions: List[str], a_primes: List[str], model: str) -> List[bool]:
    system_message = "You are an assistant helping us decide which joke is funnier given an instruction for a topic. Please respond with only \"A\" or \"B\". The empty string is not funny and neither are ungrammatical jokes. If neither joke is funny, pick one anyway."
    client = openai.AsyncOpenAI(max_retries=0)

    tasks = []
    for prompt, a, a_prime in zip(prompts, actions, a_primes):
        task = asyncio.create_task(get_winner(client, model, system_message, prompt, a, a_prime))
        tasks.append(task)

    winners = await asyncio.gather(*tasks)
    return winners


def syllable_count(word: str) -> int:
    if word.lower() in syllable_dict:
        return max([len([y for y in x if y[-1].isdigit()]) for x in syllable_dict[word.lower()]])
    else:
        # Fallback method for words not in the dictionary
        count = 0
        vowels = "aeiouy"
        word = word.lower()
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count


def get_words(line:str) -> List[str]:
    tokens = nltk.word_tokenize(line)
    return [token.lower() for token in tokens if token.isalpha()]


def is_haiku(text: str) -> bool:
    rows = text.split('\n')
    row_words = [get_words(row) for row in rows]
    row_syllables = [sum([syllable_count(word) for word in row]) for row in row_words]
    is_haiku = row_syllables == [5, 7, 5]
    return is_haiku


async def get_winners_haikus(prompts: List[str], actions: List[str], a_primes: List[str], model: str) -> List[bool]:
    system_message = "You are an assistant helping us decide which poem is better given an instruction for a topic. Please respond with only \"A\" or \"B\". We strongly prefer haikus which follow the instructions and make use of alliteration and weakly prefer haikus which use words with Latin cognates."
    client = openai.AsyncOpenAI(max_retries=0)

    tasks = []
    for prompt, a, a_prime in zip(prompts, actions, a_primes):
        a_is_haiku = is_haiku(a)
        a_prime_is_haiku = is_haiku(a_prime)
        if a_is_haiku != a_prime_is_haiku:
            task = asyncio.Future()
            task.set_result(a_is_haiku)
        else:
            task = asyncio.create_task(get_winner(client, model, system_message, prompt, a, a_prime))
        tasks.append(task)

    winners = await asyncio.gather(*tasks)
    return winners


def test_joke_winners():
    prompts = ["Tell me a joke about a dog", "tell me a joke about a cat"] * 16
    actions = ["What do you call a dog that does magic tricks? A labracadabrador.", "What do you call a cat that does magic tricks? A bad cat."] * 16
    a_primes = ["What do you call a dog that does magic tricks? A magical dog.", "What do you call a cat that does it all? Pawsome."] * 16
    for i in range(100):
        winners = asyncio.run(get_winners('jokes', prompts, actions, a_primes))
        print(winners)
        time.sleep(0.01)

def test_haiku_winners():
    prompts = ['Write a haiku containing the words "wet dog"'] * 2
    actions = ['wet dog', "Wet dog, damp and drenched,\nWanders, wistful, waterlogged,\nWhiff of wild windswept."]
    a_primes = ['Rain on fur, a splash,\nPaws dance in the puddled path,\nJoy of a wet dog.'] * 2
    a0_is_haiku = is_haiku(actions[0])
    assert not a0_is_haiku
    a0_prime_is_haiku = is_haiku(a_primes[0])
    assert a0_prime_is_haiku
    winners = asyncio.run(get_winners('haikus', prompts, actions, a_primes))
    assert winners == [False, True]
    print('Haiku test passed!')

                                                    # data = get_jokes('train')
if __name__ == '__main__':
    # test_joke_winners()
    test_haiku_winners()
