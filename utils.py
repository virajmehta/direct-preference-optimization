import os
import math
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List
from tqdm import trange
from torch import nn


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            if os.access(prefix, os.W_OK):
            # return f"{prefix}/{getpass.getuser()}"
                return f"{prefix}/dpo"
    os.makedirs(prefix)
    # return f"{prefix}/{getpass.getuser()}"
    return f"{prefix}/dpo"


def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)

class DropoutModel(nn.Module):
    def __init__(self, model, dropout, dtype):
        super(DropoutModel, self).__init__()

        self.model = model
        self.dropout = nn.Dropout(dropout).cuda()
        self.linear = nn.Linear(32000, 32000).cuda().to(dtype)
        self.config = model.config

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        dropout_output = self.dropout(output[0])

        logits = self.linear(dropout_output)

        return DropoutModelOutput(logits)

    def generate(self, inputs, attention_mask, max_length, do_sample, pad_token_id):
        return self.model.generate(inputs=inputs, attention_mask=attention_mask, max_length=max_length, do_sample=do_sample, pad_token_id=pad_token_id)

class DropoutModelOutput():
    def __init__(self, logits):
        self.logits = logits


'''
def predict_logits_with_dropout(model, input_ids, attention_mask, labels, num_samples):
    """Predict with dropout, and return the mean and variance of the predictions."""
    was_training = model.training
    model.train()
    with torch.no_grad():
        outputs = [model(input_ids, attention_mask=attention_mask) for _ in range(num_samples)]
        logits = [output.logits for output in outputs]
        logps = [_get_batch_logps(logit, labels) for logit in logits]
        predictions = torch.stack(logps)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
    if not was_training:
        model.eval()
    return mean, variance
'''


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

def predict_logits_with_dropout(model, input_ids, attention_mask, labels, num_samples, minibatch_size=32):
    """Predict with dropout, and return the mean and variance of the predictions."""
    was_training = model.training
    model.train()

    n = input_ids.size(0)
    batch_count = math.ceil(n / minibatch_size)

    logps_list = []
    with torch.no_grad():
        for batch_idx in trange(batch_count, leave=False):
            start_idx = batch_idx * minibatch_size
            end_idx = min((batch_idx + 1) * minibatch_size, n)
            input_ids_batch = input_ids[start_idx:end_idx]
            attention_mask_batch = attention_mask[start_idx:end_idx]
            labels_batch = labels[start_idx:end_idx]

            # outputs = [model(input_ids_batch, attention_mask=attention_mask_batch) for _ in range(num_samples)]
            # print('Finish inference')
            # logits = [output.logits for output in outputs]
            # logps = [_get_batch_logps(logit, labels_batch) for logit in logits]

            outputs = model(input_ids_batch.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, input_ids_batch.size(1)), attention_mask=attention_mask_batch.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, attention_mask_batch.size(1)))
            outputs.logits = outputs.logits.reshape(input_ids_batch.size(0), num_samples, input_ids_batch.size(1), -1)
            logits = [outputs.logits[:, idx,:,:].squeeze(1) for idx in range(outputs.logits.shape[1])]
            logps = [_get_batch_logps(logit, labels_batch) for logit in logits]

            logps_list.append(torch.stack(logps))

    predictions = torch.cat(logps_list, dim=1)
    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0)
    del input_ids
    del attention_mask
    del labels

    if not was_training:
        model.eval()

    return mean, variance
