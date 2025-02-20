"""
Runs BC and DAgger for hw1.
"""

import pickle
import numpy as np

import torch

import gymnasium as gym

from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure import utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.logger import Logger
from cs285.scripts.params_hw1 import *


def run_training_loop(params):
    # set seed
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    # create logger
    logger = Logger(params['logdir'])

    ### ENVIRONMENT ###
    env = gym.make(params['env_name'], render_mode='rgb_array')             # video rendering을 위해 다른 mode는 x
    _, _ = env.reset(seed=seed)                                             # reset the env (to the init ob and info)

    # max length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps   # parsing하지 않으면 env의 setting으로
    MAX_VIDEO_LEN = params['ep_len']

    # env attributes
    assert isinstance(env.action_space, gym.spaces.Box), ("Action space must be continous,"
                                                          "i.e., policy is contionous (conditional) distribution.")
    ob_dim = env.observation_space.shape[0]     # condition random variable의 차원
    ac_dim = env.action_space.shape[0]          # random variable의 차원

    # simulation timestep (video 저장에 사용)
    # simulation이 가능한 MuJoCo env라면
    if hasattr(env, 'sim') and hasattr(env.sim, 'model'):
        fps = 1 / env.sim.model.opt.timestep
    else:
        fps = env.metadata['render_fps']  # MJ_ENV_NAMES의 env들은 wrapper env가 아닌 base env이므로 env.env 불필요

    ### AGENT ###
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['hidden_size'],
        lr=params['lr']
    )

    ### EXPERT ###
    print('\nLoading expert polify from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Done restoring expert policy.')

    ### REPLAY BUFFER ###
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    ### TRAINING LOOP ###
    total_timesteps = 0
    start_time = time.time()

    for itr in range(params['n_iter']):
        print(f"\n********** Iteration {itr + 1} / {params['n_iter']} ************")

        ## Data Collection (Rolling Out)
        print(f"\nCollecting data to be used for training from... {params['expert_data']}")
        # BC training under expert data
        if itr == 0:
            trajs = pickle.load(open(params['expert_data'], 'rb'))
            print(f'Get the list of {len(trajs)} expert policies.')
            _name_rules = ['obs', 'acs', 'rwds', 'next_obs', 'dones']
            trajs = [dict(zip(_name_rules, trajs[i].values())) for i in range(len(trajs))]

            timesteps_this_batch = 0
        # DAgger training under sampled data (from policy) relabeled by expert
        else:
            assert params['do_dagger']
            # sampling transitions
            trajs, timesteps_this_batch = utils.sample_trajectories(
                env, actor, params['batch_size'], params['ep_len']
            )
            # relabelling
            print('\nRelabelling collected observations (from policy) with labels from expert policy...')
            for i in range(len(trajs)):
                trajs[i]['acs'] = expert_policy.get_action(trajs[i]['obs'])     # temporal context 고려 X

        total_timesteps += timesteps_this_batch
        replay_buffer.add_rollouts(trajs)

        ## Training Agent
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):
            # sampling transitions from replay buffer
            indices = np.random.permutation(len(replay_buffer))[:params['train_batch_size']]
            ob_batch, ac_batch = replay_buffer.observations[indices], replay_buffer.actions[indices]

            # take gradient step and log training performance
            training_log = actor.update(ob_batch, ac_batch)
            training_logs.append(training_log)

        ## Evaluation and Logging
        print('\nBeginning logging procedure...')
        log_video = ((params['video_log_freq'] != -1) and
                     (itr % params['video_log_freq'] == 0))     # 요번 iter에서 video를 rendering하고 logging할 것인지
        log_metrics = (itr % params['scalar_log_freq'] == 0)    # 요번 iter에서 eval metrics를 logging할 것인지

        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('Collecting video of evaluation rollouts...')
            eval_video_trajs = utils.sample_n_trajectories(
                env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )
            logger.log_trajs_as_videos(
                trajs=eval_video_trajs,
                step=itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title='eval_rollouts'
            )

        if log_metrics:
            # get eval metrics from (eval) transitions
            print('Collecting evaluation rollouts...')
            eval_trajs, eval_timeteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len']
            )   # buffer에서 가지고 오지 않고 새롭게 sampling
            print(f"\n--- Using approximately {round(params['eval_batch_size']/params['ep_len'])} eval batches")
            logs = utils.compute_metrics(trajs, eval_trajs)

            # compute additional metrics
            logs.update(training_logs[-1])  # 최신의 training loss 추가
            logs['Train_EnvstepsSoFar'] = total_timesteps
            logs['TimeSinceStart'] = time.time() - start_time
            if itr == 0: logs['Initial_DataCollection_AverageReturn'] = logs['Train_AverageReturn']

            # perform the logging
            for key, value in logs.items():
                print('{}: {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('--- Done logging.\n')

            logger.flush()

        if params['save_params']:
            print('\nAgent params saved.')
            filepath = str(Path(params['logdir']) / f'policy_itr_{itr}')
            actor.save(filepath)


def main(debug=False):
    if not debug:
        params = parse_args()
    else:
        params = {
            'exp_name': '_debug',
            'env_name': 'Ant-v4',
            'do_dagger': True,

            'n_layers': 2,
            'hidden_size': 32,

            'ep_len': 10,
            'batch_size': 20,
            'max_replay_buffer_size': 500,

            'n_iter': 4,
            'num_agent_train_steps_per_iter': 3,
            'train_batch_size': 10,
            'lr': 5e-3,

            'eval_batch_size': 10,

            'video_log_freq': -1,
            'scalar_log_freq': 1,
            'save_params': False,

            'seed': 1,
            'no_gpu': False,
            'which_gpu': 0
        }
    params = add_directories(params)

    run_training_loop(params)


if __name__ == "__main__":
    main()
