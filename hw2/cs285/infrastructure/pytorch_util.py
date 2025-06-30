from typing import Union
import numpy

import torch
from torch import nn


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

device = None


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    if isinstance(activation, str): activation = _str_to_activation[activation]
    if isinstance(output_activation, str): output_activation = _str_to_activation[output_activation]

    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        input_size = hidden_size
    layers.append(nn.Linear(hidden_size, output_size))
    layers.append(output_activation)

    mlp = nn.Sequential(*layers)
    mlp.to(device)

    return mlp


def init_gpu(use_gpu=True, gpu_id=0):
    global device

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("Using CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(array: numpy.ndarray):
    return torch.from_numpy(array).float().to(device)


def to_numpy(tensor: torch.Tensor):
    return tensor.to('cpu').detach().numpy()