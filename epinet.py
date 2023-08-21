import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class EpiNet(nn.Module):
    def __init__(self,
                 llm_model: nn.Module,
                 layer_sizes: List[int]=[50, 50],
                 d: int=10,
                 lambda_val: float=0.1):
        '''
        An epistemic neural net wrapper around an LLM. As in Osband et al 2021.
        llm_model: an initialized HF transformer (e.g. Llama or something)
        layer_sizes: a list of ints, the sizes of the MLP layers
        d: the dimension of the latent space
        lambda_val: the lambda value for the EpiNet
        '''
        super(EpiNet, self).__init__()
        self.base_net = llm_model
        self.lambda_val = lambda_val
        self.d = d

        # The output size of the MLPs should match the logits size
        output_size = self.base_net.config.num_labels + d

        # Define MLP g with configurable layers
        layers_g = []
        input_size = self.base_net.config.hidden_size
        for size in layer_sizes:
            layers_g.append(nn.Linear(input_size, size))
            layers_g.append(nn.ReLU())
            input_size = size
        layers_g.append(nn.Linear(input_size, output_size))
        self.g = nn.Sequential(*layers_g)

        # Define MLP h with configurable layers and make it non-trainable
        layers_h = []
        input_size = self.base_net.config.hidden_size + d
        for size in layer_sizes:
            layers_h.append(nn.Linear(input_size, size))
            layers_h.append(nn.ReLU())
            input_size = size
        layers_h.append(nn.Linear(input_size, output_size))
        self.h = nn.Sequential(*layers_h)
        for param in self.h.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                num_samples:int =1):
        # Compute phi(x) and logits
        outputs = self.base_net(input_ids=input_ids, attention_mask=attention_mask)
        # stop gradients on phi(x)
        phi_x = outputs.last_hidden_state[:, 0, :].detach()
        logits_x = outputs.logits

        # Sample z with shape (batch_size, num_samples, d)
        batch_size, _ = logits_x.size()
        z_samples = torch.randn((batch_size, num_samples, self.d), device=input_ids.device)

        # Compute g and h
        g_psi = self.g(phi_x)
        h_xi = self.h(phi_x)

        # expand phi_x to num_samples dimension
        phi_x_expanded = phi_x.unsqueeze(1).expand(-1, num_samples, -1)

        # concatenate phi_x_expanded and z_samples
        phi_x_z = torch.cat([phi_x_expanded, z_samples], dim=-1)

        # Compute g and h with concatenated phi_x and z
        # Reshape to treat multiple samples as a batch
        phi_x_z_reshaped = phi_x_z.view(-1, phi_x_z.size(-1))
        g_psi = self.g(phi_x_z_reshaped)
        h_xi = self.h(phi_x_z_reshaped)

        # Reshape back to include num_samples dimension
        g_psi = g_psi.view(batch_size, num_samples, -1)
        h_xi = h_xi.view(batch_size, num_samples, -1)

        # Final computation with broadcasting
        f_x = logits_x.unsqueeze(1) + self.lambda_val * (g_psi + h_xi)


        # Final computation
        new_outputs = outputs._replace(logits=f_x)
        return new_outputs

