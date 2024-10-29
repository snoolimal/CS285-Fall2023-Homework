"""
Set hyperparameters and parse.
"""

import argparse
from pathlib import Path
import time

MJ_ENV_NAMES = ['Ant-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Hopper-v4']


class ArgParser:
    def __init__(self):
        pass

    @staticmethod
    def parse_args():
        args = ArgParser._parse_args()
        params = vars(args)

        # logging directory
        if args.do_dagger:
            logdir_prefix = 'q2_'
            assert args.n_iter > 1, (
                "DAGGER needs more than 1 iteration (n_iter>1) of training,"
                "to iteratively query the expert and train"
                "(after 1st warmstarting from behavior cloning)."
            )  # 1st iter에서 BC로 초기 모델을 학습하고 -- i.e., 학습의 첫 단계로 BC를 통해 기본 모델 준비 -- ,
            #   그 후 DAGGER의 iteration이 본격적으로 시작
        else:
            logdir_prefix = 'q1_'
            assert args.n_iter == 1, (
                "Vanilla behavior cloning collects expert data just once (n_iter=1)."
            )

        log_path = Path.cwd().parents[0] / 'log'  # cs285/log
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=False)
        log_dir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
        log_dir = log_path / log_dir
        params['log_dir'] = log_dir
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=False)

        # expert path
        expert_pol_dir, _, expert_data_dir, _ = ArgParser.get_expert_dir()
        params['expert_pol_path'] = expert_pol_dir / f'{args.expert_policy}.pkl'
        params['expert_data_path'] = expert_data_dir / f'expert_data_dir_{args.expert_data}.pkl'

        return params

    @staticmethod
    def _parse_args():
        parser = argparse.ArgumentParser()

        # experiment
        parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
        parser.add_argument('--do_dagger', action='store_true')
        parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)

        # seed and gpu
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--no_gpu', '-ngpu', action='store_true')
        parser.add_argument('--which_gpu', type=int, default=0)

        # policy
        parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
        parser.add_argument('--hidden_size', type=int, default=64)  # width of each layer, of policy to be learned
        parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # lr for supervised learning
        parser.add_argument('--multivariate', '-mul', type=bool, action='store_true')

        # replay buffer
        parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)

        # expert policy and expert data
        _, expert_pol, _, expert_data = ArgParser.get_expert_dir()
        parser.add_argument('--expert_policy', '-epol', type=str, help=f'chocies: {", ".join(expert_pol)}', required=True)
        parser.add_argument('--expert_data', 'edata', type=str, help=f'chocies: {", ".join(expert_data)}', required=True)

        # training
        parser.add_argument('--n_iter', '-n', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=1000,
                            help='number of training data collected (in the env) during each iteration')
        parser.add_argument('--train_batch_size', type=int, default=100,
                            help='number of sampled data points to be used per gradient/train step')
        parser.add_argument('--eval_batch_size', type=int, default=1000,
                            help='eval data collected (in the env) for logging metrics')
        parser.add_argument('--ep_len', type=int,
                            help='maximum length(tsteps) of traj(episode)')
        parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000,
                            help='number of gradient steps for training policy per iter')

        # logging
        parser.add_argument('--video_log_freq', type=int, default=-1)
        parser.add_argument('--max_videos_to_save', type=int, default=2)
        parser.add_argument('--scalar_log_freq', type=int, default=1)
        parser.add_argument('--save_params', action='store_true')

        # parsing arguments
        args = parser.parse_args()

        return args

    @staticmethod
    def get_expert_dir():
        _cs_285 = Path.cwd().parent

        expert_pol_dir = _cs_285 / 'policies' / 'experts'
        expert_data_dir = _cs_285 / 'expert_data'

        expert_pol = [epol.stem for epol in expert_pol_dir.glob('*.pkl')]
        expert_data = [str(edata.stem)[len('expert_data_'):] for edata in expert_data_dir.glob('*.pkl')]

        return expert_pol_dir, expert_pol, expert_data_dir, expert_data

    @staticmethod
    def dummy_params_for_debug():
        params = {
            'exp_name': 'debug',
            'do_dagger': True,
            'env_name': 'Ant-v4',

            'seed': 42,
            'no_gpu': False,
            'which_gpu': 0,

            'n_layers': 3,
            'hidden_size': 32,
            'learning_rate': 5e-3,
            'multivariate': True,

            'max_replay_buffer_size': 1000000,

            'expert_policy': 'Ant',
            'expert_data': 'Ant-v4',

            'n_iter': 10,
            'batch_size': 32,
            'train_batch_size': 10,
            'eval_batch_size': 10,
            'ep_len': 20,
            'num_agent_train_steps_per_iter': 10,

            'video_log_freq': -1,
            'max_videos_to_save': 2,
            'scalar_log_freq': 1,
            'save_params': True
        }

        if params['do_dagger']:
            logdir_prefix = 'q2_'
            assert params['n_iter'] > 1
        else:
            logdir_prefix = 'q1_'
            assert params['n_iter'] == 1

        log_path = Path('../log')
        log_path.mkdir(parents=True, exist_ok=True)
        log_dir = params['exp_name'] + '_' + logdir_prefix + params['env_name']
        log_dir = log_path / log_dir
        params['log_dir'] = log_dir

        log_dir.mkdir(parents=True, exist_ok=True)

        expert_pol_dir, _, expert_data_dir, _ = ArgParser.get_expert_dir()
        params['expert_pol_path'] = expert_pol_dir / f"{params['expert_policy']}.pkl"
        params['expert_data_path'] = expert_data_dir / f"expert_data_{params['expert_data']}.pkl"

        return params
