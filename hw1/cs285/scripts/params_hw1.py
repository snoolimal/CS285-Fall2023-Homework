"""
Set hyperparameters and parse.
"""

import argparse
import time
from pathlib import Path

MAX_NVIDEO = 2      # tensorboard에 video로 저장할 rollouts의 수
MAX_VIDEO_LEN = 40  # training loop에서 덮어 쓸 것

MJ_ENV_NAMES = ['Ant-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Hopper-v4']


def parse_args():
    parser = argparse.ArgumentParser()

    # experiment ID
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--do_dagger', action='store_true')

    # universal arguments
    parser.add_argument('--n_iter', '-n', type=int, default=1,
                        help='1 for Vanila BC, otherwise num of DAgger iteration')
    parser.add_argument('--ep_len', type=int)  # (sampling할) trajectory의 길이 제한

    # rolling out (data collection)
    parser.add_argument('--batch_size', type=int, default=1000)     # 전체 (rollouts의) transitions의 수
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)

    # training
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)     # iter마다 사용할 batch의 수
    parser.add_argument('--train_batch_size', type=int, default=100)

    # validation
    parser.add_argument('--eval_batch_size', type=int, default=1000)

    # policy
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lr', '-lr', type=float, default=5e-3)

    # logging
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--video_log_freq', type=int, default=5, help='-1 for not logging video')
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    # seed and gpu
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    return params


def add_directories(params):
    # get expert-related directories
    _cs285 = Path(__file__).resolve().parents[1]
    _expert_policy = params['env_name'].split('-')[0] + '.pkl'

    params['expert_policy_file'] = str(_cs285 / 'policies' / 'experts' / _expert_policy)
    params['expert_data'] = str(_cs285 / 'expert_data' / f"expert_data_{params['env_name']}.pkl")

    # create log directory
    if params['do_dagger']:
        logdir_prefix = 'q2_'
        assert params['n_iter'] > 1, ("DAgger needs more than 1 iteration, i.e., n_iter>1, of training,"
                                      "to iteratively query the expert and train (after 1st warmstarting from BC).")
    else:
        logdir_prefix = 'q1_'
        assert params['n_iter'] == 1, "Vanilla behavior cloning collects expert data just once (n_iter=1)."
    data_path = _cs285 / 'data'
    data_path.mkdir(exist_ok=True)
    logdir = logdir_prefix + params['exp_name'] + '_' + params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = data_path / logdir
    logdir.mkdir(exist_ok=True)
    params['logdir'] = str(logdir)

    return params
