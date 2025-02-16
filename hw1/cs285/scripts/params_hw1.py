"""
Set hyperparameters and parse.
"""

import argparse
import time
from pathlib import Path


MAX_NVIDEO = 2      # tensorboard에 video로 저장할 rollouts의 수
MAX_VIDEO_LEN = 40  # training loop에서 덮어 쓸 것

MJ_ENV_NAMES = ['Ant-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Hopper-v4']


def parse_args(debug=False):
    parser = argparse.ArgumentParser()

    # expert loading (이 script를 실행하는 dir에 따라 설정)
    # parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)
    # parser.add_argument('--expert_data', '-ed', type=str, required=True)

    # experiment
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--do_dagger', action='store_true')

    # policy network
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)

    # rolling out
    parser.add_argument('--ep_len', type=int)
    parser.add_argument('--batch_size', type=int, default=1000)  # 각 iter에서 sampling할 transitions의 수
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)

    # training
    parser.add_argument('--n_iter', '-n', type=int, default=1)  # vanilla BC라면 1, DAgger라면 DAgger iteration 횟수
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # 각 iter에서 밟을 grad steps의 수
    parser.add_argument('--train_batch_size', type=int, default=100)  # 각 grad step에 사용하는 (training) trainsitions의 수
    parser.add_argument('--lr', '-lr', type=float, default=5e-3)

    # validation
    parser.add_argument('--eval_batch_size', type=int, default=1000)  # eval metrics log를 찍을 eval transitions의 수

    # logging
    parser.add_argument('--video_log_freq', type=int, default=5, help='-1 for not logging video')
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true')

    # seed and gpu
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)

    args = parser.parse_args()
    params = vars(args)         # convert args to dict

    # create directory for logging
    if args.do_dagger:
        logdir_prefix = 'q2_'
        assert args.n_iter > 1, ("DAgger needs more than 1 iteration, i.e., n_iter>1, of training,"
                                 "to iteratively query the expert and train (after 1st warmstarting from BC).")
    else:
        logdir_prefix = 'q1_'
        assert args.n_iter == 1, "Vanilla behavior cloning collects expert data just once (n_iter=1)."
    _cs285 = Path(__file__).resolve().parents[1]
    data_path = _cs285 / 'data'
    data_path.mkdir(exist_ok=True)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = data_path / logdir
    params['logdir'] = str(logdir)
    logdir.mkdir(exist_ok=True)

    if debug:
        params = {
            'exp_name': '_debug',
            'env_name': 'Ant-v4',
            'do_dagger': True,

            'n_layers': 2,
            'hidden_size': 64,

            'ep_len': None,
            'batch_size': 1000,
            'max_replay_buffer_size': 1000000,

            'n_iter': 2,
            'num_agent_train_steps_per_iter': 1000,
            'train_batch_size': 100,
            'lr': 5e-3,

            'eval_batch_size': 1000,

            'video_log_freq': -1,
            'scalar_log_freq': 1,
            'save_params': False,

            'seed': 1,
            'no_gpu': False,
            'which_gpu': 0
        }

    return params
