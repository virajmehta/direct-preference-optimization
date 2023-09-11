import openai
import os
from dotenv import load_dotenv
import numpy as np
from typing import List, Optional, Tuple
import asyncio
from asyncio import Semaphore
import logging
from time import time, sleep


if __name__ == '__main__':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)


openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logging.warning("openai.api_key is None")

start = None
num_requests = 0

class TokenBucket:
    def __init__(self, rate: int):
        # rate is in requests per second
        self._rate = rate
        self._capacity = rate
        self._tokens = self._capacity
        self._last_refill = time()

    async def consume(self):
        while self._tokens < 1:
            self._refill()
            await asyncio.sleep(1)  # Sleep for some time before trying again
        self._tokens -= 1
        global num_requests
        num_requests += 1

    def _refill(self):
        now = time()
        time_passed = now - self._last_refill
        refill_amount = time_passed * self._rate
        self._tokens = min(self._capacity, self._tokens + refill_amount)
        self._last_refill = now

MaybeTokenBucket = Optional[TokenBucket]


async def _call_chat(system_prompt: str, user_prompt:str, token_bucket: MaybeTokenBucket=None, model="gpt-3.5-turbo") -> str:
    done = False
    messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    while not done:
        try:
            if token_bucket is not None:
                await token_bucket.consume()
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                )
            completion = response.choices[0].message.content
            done=True
        except Exception as e:
            if token_bucket is None:
                sleep(1)
    return completion


async def _handle_chat(system_prompt: str, user_prompt: str, token_bucket: TokenBucket, semaphore: Semaphore, model: str) -> str:
    async with semaphore:
        completion = await _call_chat(system_prompt, user_prompt, token_bucket)
    global num_requests
    if num_requests % 1000 == 0:
        duration = time() - start
        duration_min = duration / 60
        print(f"{num_requests=}, {duration=:.2f} rate per min={num_requests / duration_min:.2f}")
    return completion


def call_chats(prompts: List[Tuple[str, str]], model: str="gpt-3.5-turbo") -> List[str]:
    # prompts should be [(system_prompt, user_prompt), ...]
    global start
    start = time()
    global num_requests
    num_requests = 0
    max_concurrent_tasks = 20
    oai_quotas = {'gpt-3.5-turbo': 3500, 'gpt-4': 200}
    oai_quota_per_minute = oai_quotas[model]
    oai_quota_per_second = oai_quota_per_minute // 60
    semaphore = Semaphore(max_concurrent_tasks)
    token_bucket = TokenBucket(oai_quota_per_second)
    async def gather_tasks():
        tasks = [_handle_chat(system_prompt, user_prompt, token_bucket, semaphore, model) for system_prompt, user_prompt in prompts]
        return await asyncio.gather(*tasks)
    return asyncio.run(gather_tasks())


def test_chats():
    fun_sentences = [
    "The sky is blue today.",
    "Ducks quack to communicate.",
    "Bananas are my favorite fruit.",
    "Chocolate makes everything better.",
    "Singing in the rain is fun.",
    "Cats have nine lives, they say.",
    "The moon is made of cheese.",
    "Robots will take over the world.",
    "Pineapples belong on pizza.",
    "Unicorns are just horses with a twist."
    ]
    system_prompts = ["You are a pig-latinifiying bot. Please reproduce the user message in Pig Latin"] * len(fun_sentences)

    completions = call_chats(list(zip(system_prompts, fun_sentences)))
    for completion, fun_sentence in zip(completions, fun_sentences):
        print(f"{fun_sentence=}, {completion=}")


if __name__ == '__main__':
    test_chats()
