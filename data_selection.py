import torch
import datasets
import random
import time
import numpy as np
from typing import List, Optional, Iterator, Dict
from utils import TemporarilySeededRandom, predict_logits_with_dropout
from preference_datasets import get_dataset, get_collate_fn, tokenize_batch_element

pretrain_fraction = 0.4 # don't mess with this too much, here in theory 1 would be all SFT and 0 would be all RLHF


def get_shuffle_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       pretrain_mode: bool = False,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       **kwargs) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        pretrain_mode: Whether to use the pretraining fraction of the dataset.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"


    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir).items():
                this_flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
            if split == 'train':
                split_idx = int(pretrain_fraction * len(this_flat_data))
                if pretrain_mode:
                    this_flat_data = this_flat_data[:split_idx]
                else:
                    this_flat_data = this_flat_data[split_idx:]
            flat_data.extend(this_flat_data)

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    is_train = split == 'train'
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
                    if not is_train:
                        break
        if done:
            break

        epoch_idx += 1


def get_active_iterator(names: List[str],
                        tokenizer,
                        split: str = 'train',
                        batch_size: int = 1,
                        selection_ratio: float = 3.,
                        shuffle: bool = True,
                        max_length: int = 512,
                        max_prompt_length: int = 128,
                        pretrain_mode: bool = False,
                        sft_mode: bool = False,
                        n_epochs: Optional[int] = None,
                        n_examples: Optional[int] = None,
                        seed:int = 0,
                        silent: bool = False,
                        cache_dir: Optional[str] = None,
                        policy: Optional[torch.nn.Module] = None,
                        ref_policy: Optional[torch.nn.Module] = None,
                        n_samples: int = 5,
                        selection_strategy:str = 'ae',  # 'ae' or 'us'
                        beta: float = 2.,
                        **kwargs) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        pretrain_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        policy: pointer to current model
        ref_policy: pointer to reference model
        n_samples: number of samples to draw from the policy for uncertainty estimation
        selection_strategy: 'ae' or 'us' for active exploration or uncertainty sampling
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """
    assert not sft_mode, "Active iterator should never be used for SFT" # TODO: maybe we might want it for a comparison later, but this is the assumption today
    assert not pretrain_mode, "Active iterator should never be used for pretraining" # TODO: maybe we might want it for a comparison later, but this is the assumption today
    # assert n_examples is not None, "Must specify n_examples for this"
    assert policy is not None, "need a model for the active iterator"



    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir).items():
                this_flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
            if split == 'train':
                split_idx = int(pretrain_fraction * len(this_flat_data))
                if pretrain_mode:
                    this_flat_data = this_flat_data[:split_idx]
                else:
                    this_flat_data = this_flat_data[split_idx:]
            flat_data.extend(this_flat_data)

    # should now have flat_data = [(prompt, responses, pairs, sft_target, truncation_mode), ...]

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    # here is where we need to get this down to batch size
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size * selection_ratio:
                        collated_batch = collate_fn(batch)
                        if selection_strategy == 'ae':
                            selected_batch = select_best_elements(batch=collated_batch,
                                                                  num_to_select=batch_size,
                                                                  policy=policy,
                                                                  ref_policy=ref_policy,
                                                                  n_samples=n_samples,
                                                                  beta=beta)
                        elif selection_strategy == 'us':
                            selected_batch = select_us_elements(batch=collated_batch,
                                                                num_to_select=batch_size,
                                                                policy=policy,
                                                                ref_policy=ref_policy,
                                                                n_samples=n_samples)
                        else:
                            raise NotImplementedError(f'Selection strategy {selection_strategy} not implemented')
                        yield selected_batch
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def compute_means_variances(a1_input_ids: torch.Tensor,
                            a1_attention_mask: torch.Tensor,
                            a1_labels: torch.Tensor,
                            a2_input_ids: torch.Tensor,
                            a2_attention_mask: torch.Tensor,
                            a2_labels: torch.Tensor,
                            policy: torch.nn.Module,
                            n_samples: int):
    ga1_mean, ga1_variance = predict_logits_with_dropout(policy, a1_input_ids, a1_attention_mask, a1_labels, n_samples)
    a1_mean = ga1_mean.to('cpu').float()
    a1_variance = ga1_variance.to('cpu').float()
    del ga1_mean
    del ga1_variance
    ga2_mean, ga2_variance = predict_logits_with_dropout(policy, a2_input_ids, a2_attention_mask, a2_labels, n_samples)
    a2_mean = ga2_mean.to('cpu').float()
    a2_variance = ga2_variance.to('cpu').float()
    del ga2_mean
    del ga2_variance
    return a1_mean, a1_variance, a2_mean, a2_variance


def compute_ae_uncertainty(batch: List[Dict],
                           policy: torch.nn.Module,
                           ref_policy: torch.nn.Module,
                           n_samples: int,
                           beta: float):
    device = next(policy.parameters()).device
    a1_input_ids = batch['chosen_input_ids'].to(device)
    a1_attention_mask = batch['chosen_attention_mask'].to(device)
    a1_labels = batch['chosen_labels'].to(device)
    a2_input_ids = batch['rejected_input_ids'].to(device)
    a2_attention_mask = batch['rejected_attention_mask'].to(device)
    a2_labels = batch['rejected_labels'].to(device)
    a1_mean, a1_variance, a2_mean, a2_variance = compute_means_variances(a1_input_ids,
                                                                         a1_attention_mask,
                                                                         a1_labels,
                                                                         a2_input_ids,
                                                                         a2_attention_mask,
                                                                         a2_labels,
                                                                         policy, n_samples)
    gref_logits_a1, todel1 = predict_logits_with_dropout(ref_policy, a1_input_ids, a1_attention_mask, a1_labels, 1)
    ref_logits_a1 = gref_logits_a1.to('cpu').float()
    del gref_logits_a1
    del todel1
    gref_logits_a2, todel2 = predict_logits_with_dropout(ref_policy, a2_input_ids, a2_attention_mask, a2_labels, 1)
    ref_logits_a2 = gref_logits_a2.to('cpu').float()
    del gref_logits_a2
    del todel2
    a1_std = torch.sqrt(a1_variance)
    a2_std = torch.sqrt(a2_variance)
    upper_bounds = torch.max(a1_mean + beta * a1_std - ref_logits_a1, a2_mean + beta * a2_std - ref_logits_a2)
    lower_bounds = torch.max(a1_mean - beta * a1_std - ref_logits_a1, a2_mean - beta * a2_std - ref_logits_a2)
    uncertainties = upper_bounds - lower_bounds
    del a1_input_ids
    del a1_attention_mask
    del a1_labels
    del a2_input_ids
    del a2_attention_mask
    del a2_labels
    return uncertainties


def compute_us_uncertainty(batch: List[Dict],
                           policy: torch.nn.Module,
                           n_samples: int,
                           **kwargs):
    device = next(policy.parameters()).device
    a1_input_ids = batch['chosen_input_ids'].to(device)
    a1_attention_mask = batch['chosen_attention_mask'].to(device)
    a1_labels = batch['chosen_labels'].to(device)
    a2_input_ids = batch['rejected_input_ids'].to(device)
    a2_attention_mask = batch['rejected_attention_mask'].to(device)
    a2_labels = batch['rejected_labels'].to(device)
    a1_mean, a1_variance, a2_mean, a2_variance = compute_means_variances(a1_input_ids,
                                                                         a1_attention_mask,
                                                                         a1_labels,
                                                                         a2_input_ids,
                                                                         a2_attention_mask,
                                                                         a2_labels,
                                                                         policy, n_samples)
    del a1_input_ids
    del a1_attention_mask
    del a1_labels
    del a2_input_ids
    del a2_attention_mask
    del a2_labels
    uncertainties = (a1_variance + a2_variance) / 2
    return uncertainties


def select_best_elements(batch: List[Dict],
                         num_to_select: int,
                         policy: torch.nn.Module,
                         ref_policy: torch.nn.Module,
                         n_samples: int,
                         beta: float = 2.):
    # mean, variance = predict_logits_with_dropout(policy, input_ids, attention_mask, labels, 5)
    # don't use the fact that one is chosen or not
    start_time = time.time()
    uncertainties = compute_ae_uncertainty(batch, policy, ref_policy, n_samples, beta)
    values, indices = torch.topk(uncertainties, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    end_time = time.time()
    torch.cuda.empty_cache()
    print(f"Data selection elapsed: {end_time - start_time:.2f}s")
    return out_batch


def select_us_elements(batch: List[Dict],
                       num_to_select: int,
                       policy: torch.nn.Module,
                       ref_policy: torch.nn.Module,
                       n_samples: int,
                       ):
    # mean, variance = predict_logits_with_dropout(policy, input_ids, attention_mask, labels, 5)
    # don't use the fact that one is chosen or not
    start_time = time.time()
    uncertainties = compute_us_uncertainty(batch, policy, n_samples)
    values, indices = torch.topk(uncertainties, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    end_time = time.time()
    torch.cuda.empty_cache()
    print(f"Data selection elapsed: {end_time - start_time:.2f}s")
    return out_batch
