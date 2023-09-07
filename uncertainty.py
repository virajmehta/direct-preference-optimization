import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, disable_dropout, DropoutModel
from epinet import EpiNet, EpiNetConfig
import os
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket

import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Union, Tuple

from utils import get_local_dir, rank0_print
from data_selection import get_shuffle_iterator
import math

OmegaConf.register_new_resolver("get_local_run_dir",
                                lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

def predict_logits_with_variance(model, input_ids, attention_mask, labels, num_samples, minibatch_size=1, dropout=False, average_logprob=False):
    """Predict with dropout, and return the mean and variance of the predictions."""
    if dropout:
        was_training = model.training
        model.train()

    n = input_ids.size(0)
    batch_count = math.ceil(n / minibatch_size)
    # print(f"batch_count: {batch_count}")

    logps_list = []
    with torch.no_grad():
        for batch_idx in range(batch_count):
            # print(f"batch_idx: {batch_idx}")
            start_idx = batch_idx * minibatch_size
            end_idx = min((batch_idx + 1) * minibatch_size, n)
            input_ids_batch = input_ids[start_idx:end_idx]
            attention_mask_batch = attention_mask[start_idx:end_idx]
            labels_batch = labels[start_idx:end_idx]
            # print("Starting inference")

            # outputs = [model(input_ids_batch, attention_mask=attention_mask_batch) for _ in range(num_samples)]
            # print('Finish inference')
            # logits = [output.logits for output in outputs]
            # logps = [_get_batch_logps(logit, labels_batch) for logit in logits]

            outputs = model(input_ids_batch.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, input_ids_batch.size(1)),
                            attention_mask=attention_mask_batch.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1,
                                                                                                               attention_mask_batch.size(
                                                                                                                   1)))
            # print('Finish inference')
            outputs.logits = outputs.logits.reshape(input_ids_batch.size(0), num_samples, input_ids_batch.size(1), -1)
            logits = [outputs.logits[:, idx, :, :].squeeze(1) for idx in range(outputs.logits.shape[1])]
            logps = [_get_batch_logps(logit, labels_batch, average_log_prob=average_logprob) for logit in logits]

            logps_list.append(torch.stack(logps))

    predictions = torch.cat(logps_list, dim=1)
    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    del input_ids
    del attention_mask
    del labels

    if dropout:
        if not was_training:
            model.eval()
    return mean, variance

    return mean, variance


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


class Evaluator(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str,
                 reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.policy = policy
        self.reference_model = reference_model

    def get_train_eval_iterators(self, config: DictConfig, dataset: str, tokenizer):
        data_iterator_kwargs = dict(
            names=[dataset],
            tokenizer=tokenizer,
            shuffle=False,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        rank = 0
        train_iterator = get_shuffle_iterator(**data_iterator_kwargs, split='train', n_epochs=1,
                                            n_examples=config.n_examples, batch_size=config.batch_size,
                                            silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')
        eval_iterator = get_shuffle_iterator(**data_iterator_kwargs, split='test', n_epochs=1,
                                           n_examples=config.n_examples,
                                           batch_size=config.batch_size, silent=rank != 0,
                                           cache_dir=get_local_dir(config.local_dirs))
        return train_iterator, eval_iterator

    def predict_uncertainty(self, iterator, dropout, average_logprob):
        variances = []
        it_idx = 1
        for batch in iterator:
            print(f'Predicting for batch {it_idx * self.config.batch_size}')
            it_idx += 1
            input_ids = batch['chosen_input_ids']
            attention_mask = batch['chosen_attention_mask']
            labels = batch['chosen_labels']
            mean, variance = predict_logits_with_variance(self.policy, input_ids, attention_mask, labels, self.config.num_samples, dropout, average_logprob)
            # print(variance.shape)
            variance = variance.cpu().numpy().tolist()
            variances.extend(variance)
            print(f'Variance: {variance}')
        return variances


    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        policy_output = self.policy.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length,
            do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        if self.config.loss.name == 'dpo':
            reference_output = self.reference_model.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name == 'dpo':
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def evaluate(self, iterator):
        variances = []
        for batch in iterator:
            input_ids = batch['chosen_input_ids']
            attention_mask = batch['chosen_attention_mask']
            labels = batch['chosen_labels']
            _, variance = predict_logits_with_variance(self.policy, input_ids, attention_mask, labels, config.num_samples)
            variance.extend(variance.flatten().tolist())
        return variances


@hydra.main(version_base=None, config_path="config", config_name="config_uncertainty")
def main(config: DictConfig):
    print('Before')
    model_kwargs = {'device_map': 'auto'}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    print('policy_dtype', policy_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True,
        # torch_dtype=policy_dtype,
        quantization_config=bnb_config,
        output_hidden_states=True,
        **model_kwargs)
    # policy.config.dtype = torch.bfloat16
    print(policy)
    # if not config.dropout:
    #     disable_dropout(policy)
    policy.gradient_checkpointing_enable()
    policy = prepare_model_for_kbit_training(policy)

    if 'pythia' in config.model.name_or_path:
        target_modules = ['query_key_value']
    elif 'llama' in config.model.name_or_path:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    loraconfig = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    policy = get_peft_model(policy, loraconfig)
    for name, module in policy.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if hasattr(module, 'weight'):
            if module.weight.dtype == torch.float32:
                module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
        # print(name, module.dtype)
    if config.epinet:
        epinet_config = EpiNetConfig(lambda_val=config.lambda_val)
        policy = EpiNet(epinet_config, policy)

    if config.have_llm_dropout:
        policy = DropoutModel(policy, config.llm_dropout)
    state = torch.load(f'{get_local_dir(config.local_dirs)}/{config.model_dir}/policy.pt', map_location='cuda:0')['state']
    policy.load_state_dict(state)
    evaluator = Evaluator(policy, config, config.seed, None)

    tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
    rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                                                cache_dir=get_local_dir(config.local_dirs))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = ['shp', 'jeopardy']
    # datasets = ['jeopardy']
    datasets_compare = [dataset for dataset in datasets if dataset not in config.datasets]
    reference_train_it, reference_eval_it = evaluator.get_train_eval_iterators(config, config.datasets[0], tokenizer)
    compare_its = []
    for dataset in datasets_compare:
        compare_it_train, compare_it_eval = evaluator.get_train_eval_iterators(config, dataset, tokenizer)
        compare_its.append(compare_it_train)

    train_uncertainties = evaluator.predict_uncertainty(reference_train_it, dropout=config.dropout, average_logprob=config.average_logprob)
    eval_uncertainties = evaluator.predict_uncertainty(reference_eval_it, dropout=config.dropout, average_logprob=config.average_logprob)
    with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_{config.datasets[0]}_{config.datasets[0]}_train_{config.lora_dropout}_{config.llm_dropout}_dropout_full.pkl', 'wb') as f:
        pickle.dump(train_uncertainties, f)
    with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_{config.datasets[0]}_{config.datasets[0]}_eval_{config.lora_dropout}_{config.llm_dropout}_dropout_full.pkl', 'wb') as f:
        pickle.dump(eval_uncertainties, f)
    compare_uncertainties = []
    for compare_it, ds_name in zip(compare_its, datasets_compare):
        compare_uncertainties.append(evaluator.predict_uncertainty(compare_it, dropout=config.dropout, average_logprob=config.average_logprob))
        with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_{config.datasets[0]}_{ds_name}_{config.lora_dropout}_{config.llm_dropout}_dropout_full.pkl', 'wb') as f:
            pickle.dump(compare_uncertainties[-1], f)
    #
    sns.kdeplot(train_uncertainties, label=f'train {config.datasets[0]}')
    sns.kdeplot(eval_uncertainties, label=f'eval {config.datasets[0]}')
    for i, compare_uncertainty in enumerate(compare_uncertainties):
        sns.kdeplot(compare_uncertainty, label=f'{datasets_compare[i]}')
    plt.legend()
    # plt.savefig(f'uncertainty_epinet_{config.datasets[0]}_{config.lambda_val}.png')
    plt.savefig(f'uncertainty_{config.datasets[0]}_{config.lora_dropout}_{config.llm_dropout}_dropout_full.png')


if __name__ == '__main__':
    main()