import pickle

import numpy as np

import torch
from torch import nn

from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import pytorch_util as ptu


def create_linear_layer(W, b) -> nn.Linear:
    out_features, in_features = W.shape
    linear_layer = nn.Linear(in_features, out_features)

    linear_layer.weight.data = ptu.from_numpy(W.T)
    linear_layer.bias.data = ptu.from_numpy(b[0])

    return linear_layer


def read_layer(_layer):
    assert list(_layer.keys()) == ['AffineLayer']
    assert sorted(_layer['AffineLayer'].keys()) == ['W', 'b']

    return _layer['AffineLayer']['W'].astype(np.float32), _layer['AffineLayer']['b'].astype(np.float32)


class LoadedGaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, filename, **kwargs):
        super(LoadedGaussianPolicy, self).__init__(**kwargs)

        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        self.nonlin_type = data['nonlin_type']
        if self.nonlin_type == 'lrelu':
            self.non_lin = nn.LeakyReLU(0.01)
        elif self.nonlin_type == 'tanh':
            self.non_lin = nn.Tanh()
        else:
            raise NotImplementedError()
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', ('Policy type {} not supported'.format(policy_type))
        self.policy_params = data[policy_type]

        assert set(self.policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        # Build the policy
        # 1. Statistics for normalize input
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        self.ob_dim = obsnorm_mean.shape[-1]
        # print(f'obsnorm_mean_shape: {obsnorm_mean.shape}\n'
        #       f'obsnorm_std_shape: {obsnorm_stdev.shape}')

        self.obs_norm_mean = nn.Parameter(ptu.from_numpy(obsnorm_mean))
        self.obs_norm_std = nn.Parameter(ptu.from_numpy(obsnorm_stdev))

        # 2. Hidden layer
        self.hidden_layers = nn.ModuleList()

        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']

        for layer_name in sorted(layer_params.keys()):
            _layer = layer_params[layer_name]
            W, b = read_layer(_layer)
            linear_layer = create_linear_layer(W, b)
            self.hidden_layers.append(linear_layer)

        # 3. Output layer
        W, b = read_layer(self.policy_params['out'])
        self.output_layer = create_linear_layer(W, b)

    def forward(self, observations):
        # truncate or pad the observation if it does not match.
        if self.ob_dim != observations.shape[-1]:
            n_pad = self.ob_dim - observations.shape[-1]
            if n_pad > 0:
                observations = nn.functional.pad(observations, (0, n_pad), 'constant')
            else:
                observations = observations[:, :self.ob_dim]

        # normalization
        normed_observations = (observations - self.obs_norm_mean) / (self.obs_norm_std + 1e-6)

        h = normed_observations
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.non_lin(h)

        return self.output_layer(h)

    def update(self, normed_observations, actions_na, adv_n=None, actions_labels_na=None):
        raise RuntimeError('This policy class simply loads in a particular type of policy and queries it.'
                           'Do not try to train it.')

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        observations = ob[None, :] if len(ob.shape) == 1 else ob  # convert to batch form to for forward() call
        observations = ptu.from_numpy(observations.astype(np.float32))
        actions = self(observations)   # nn.Module의 instance는 __call__()을 override하여 forward()를 호출

        return ptu.to_numpy(actions)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)