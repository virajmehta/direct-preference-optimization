import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, DropoutModel
import os
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
from epinet import EpiNet, EpiNetConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set

import pickle

OmegaConf.register_new_resolver("get_local_run_dir",
                                lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

torch.set_default_dtype(torch.float16)

# Sleep for 8h
# import time
# time.sleep(8 * 60 * 60)


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module,
                reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model,
                           rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)

    print('building policy')
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
        # if 'lm_head' in name:
        #     module.training = True
        #     module.weight.requires_grad = True
        # if hasattr(module, 'weight'):
        #     print(name, module.weight.requires_grad)
        # print(name, module.training)
        # print(name, module.dtype)
    if config.epinet:
        epinet_config = EpiNetConfig(lambda_val=config.lambda_val)
        policy = EpiNet(epinet_config, policy)

    if config.have_llm_dropout:
        policy = DropoutModel(policy, config.llm_dropout)
    print(policy)
    # Print dtypes of all layers in policy
    # for name, module in policy.named_modules():
    #     if hasattr(module, 'weight'):
    #         print(name, module.weight.dtype)
    #     elif hasattr(module, 'dtype'):
    #         print(name, module.dtype)
    #     else:
    #         print(name, 'no dtype')


    if config.loss.name == 'dpo':
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            output_hidden_states=True,
            **model_kwargs)
        reference_model.gradient_checkpointing_enable()
        reference_model = prepare_model_for_kbit_training(reference_model)
        reference_model = get_peft_model(reference_model, loraconfig)
        for name, module in reference_model.named_modules():
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

        if config.epinet:
            epinet_config = EpiNetConfig(lambda_val=config.lambda_val)
            reference_model = EpiNet(epinet_config, reference_model)
        if config.have_llm_dropout:
            reference_model = DropoutModel(reference_model, 0.)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(
            f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name == 'dpo':
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    if 'FSDP' in config.trainer:
        raise NotImplementedError("Lora + FSDP doesn't work yet")
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
