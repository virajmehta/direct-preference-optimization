import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from typing import List


class EpiNetConfig:
    def __init__(self,
                 layer_sizes: List[int]=[50, 50],
                 d: int=10,
                 lambda_val: float=0.1):
        self.layer_sizes = layer_sizes
        self.d = d
        self.lambda_val = lambda_val


class EpiNet(PreTrainedModel):
    def __init__(self,
                 config,
                 llm: nn.Module,
        '''
        An epistemic neural net wrapper around an LLM. As in Osband et al 2021.
        llm: an initialized HF transformer (e.g. Llama or something)
        layer_sizes: a list of ints, the sizes of the MLP layers
        d: the dimension of the latent space
        lambda_val: the lambda value for the EpiNet
        '''
        super(EpiNet, self).__init__()
        self.base_net = llm
        self.config = config
        self.layer_sizes = config.layer_sizes
        self.d = config.d
        self.lambda_val = config.lambda_val

        # The output size of the MLPs should match the logits size
        output_size = self.base_net.config.vocab_size

        # Define MLP g with configurable layers
        layers_g = []
        input_size = self.base_net.config.hidden_size + d
        for size in layer_sizes:
            layers_g.append(nn.Linear(input_size, size))
            layers_g.append(nn.ReLU())
            input_size = size
        layers_g.append(nn.Linear(input_size, output_size))
        self.g = nn.Sequential(*layers_g).to(self.base_net.device)

        # Define MLP h with configurable layers and make it non-trainable
        layers_h = []
        input_size = self.base_net.config.hidden_size + d
        for size in layer_sizes:
            layers_h.append(nn.Linear(input_size, size))
            layers_h.append(nn.ReLU())
            input_size = size
        layers_h.append(nn.Linear(input_size, output_size))
        self.h = nn.Sequential(*layers_h).to(self.base_net.device)
        for param in self.h.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                num_samples:int =1):
        # Compute phi(x) and logits
        outputs = self.base_net(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # stop gradients on phi(x)
        # batch x sequence_length x hidden_size
        phi_x = outputs.hidden_states[-1].detach()
        batch_size, sequence_length, hidden_size = phi_x.size()
        phi_x_flattened = phi_x.view(-1, hidden_size)
        # batch x sequence_length x vocab_size
        logits_x = outputs.logits
        vocab_size = logits_x.shape[-1]
        logits_x_flattened = logits_x.view(-1, vocab_size)

        # Sample z with shape (batch_size, num_samples, d)
        z_samples = torch.randn((batch_size * sequence_length, num_samples, self.d), device=input_ids.device)

        # expand phi_x to num_samples dimension
        phi_x_expanded = phi_x_flattened.unsqueeze(1).expand(-1, num_samples, -1)

        # concatenate phi_x_expanded and z_samples
        phi_x_z = torch.cat([phi_x_expanded, z_samples], dim=-1)

        # Compute g and h with concatenated phi_x and z
        # Reshape to treat multiple samples as a batch
        phi_x_z_reshaped = phi_x_z.view(-1, phi_x_z.size(-1))
        g_psi = self.g(phi_x_z_reshaped)
        h_xi = self.h(phi_x_z_reshaped)

        # Reshape back to include num_samples dimension
        g_psi = g_psi.view(batch_size, sequence_length, num_samples, -1)
        h_xi = h_xi.view(batch_size, sequence_length, num_samples, -1)
        logits_x_expanded = logits_x_flattened.unsqueeze(1).view(batch_size, sequence_length, num_samples, -1)

        # Final computation with broadcasting
        # batch_size x sequence_length x num_samples x vocab_size
        f_x = logits_x_expanded + self.lambda_val * (g_psi + h_xi)
        if num_samples == 1:
            f_x = f_x[:, :, 0, :]

        # Final computation
        outputs['logits'] = f_x
        return outputs

