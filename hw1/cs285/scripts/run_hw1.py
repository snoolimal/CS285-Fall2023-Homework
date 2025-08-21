"""
Run BC or DAgger for homework 1.
"""


import pickle
import time
import pathlib
import numpy as np

import torch

import gymnasium as gym

from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure import utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.logger import Logger


MAX_NVIDEO = 2  # how many rollouts to save as videos to tensorboard
MAX_VIDEO_LEN = 40  # we overwrite this in the training loop

MJ_ENV_NAMES = ['Ant-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Hopper-v4']


def run_training_loop(params):
    """
    Runs training with the specified parameters (behavior cloning or DAgger).
    """
    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    # Get logger
    logger = Logger(params['logdir'])

    # ENV
    # make the gym environment
    env = gym.make(params['env_name'], render_mode='rgb_array')     # video rendering을 위해 다른 mode는 X
    _, _ = env.reset(seed=seed)

    # maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    # env attributes
    assert isinstance(env.action_space, gym.spaces.Box), 'Environment must be continuous.'
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    # simulation이 가능한 env라면
    if hasattr(env, 'sim') and hasattr(env.sim, 'model'):
        fps = 1 / env.sim.model.opt.timestep
    else:
        fps = env.metadata['render_fps']  # MJ_ENV_NAMES의 env들은 wrapper env가 아닌 base env이므로 env.env 불필요

    # AGENT
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
    )

    # REPLAY BUFFER
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    # LOAD EXPERT POLICY
    print('\nLoading expert polify from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Done restoring expert policy...')

    # TRAINING LOOP
    # init vars at beginning of training
    total_envsteps = 0
    start_time = time.time()

    for itr in range(params['n_iter']):
        print(f"\n********** Iteration {itr + 1} / {params['n_iter']} **********")

        print('\nCollecting data to be used for training...')
        if itr == 0:
            # BC training from expert data
            trajs = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else:
            # DAgger training from sampled data relabeled by expert
            assert params['do_dagger']
            # sampling
            trajs, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'])
            # relabel the collected observations with actions from provided expert policy
            print("\nRelabelling collected observations with labels from an expert policy...")
            for i in range(len(trajs)):     # i는 항상 0
                trajs[i]['action'] = expert_policy.get_action(trajs[i]['observation'])

        total_envsteps += envsteps_this_batch
        replay_buffer.add_rollouts(trajs)   # add collected data to replay buffer

        # train agent (using sampled data from replay buffer)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):
            # sample transitions from replay buffer
            # for imitation learning, we only need observations and actions
            indices = np.random.permutation(len(replay_buffer))[:params['train_batch_size']]
            ob_batch, ac_batch = replay_buffer.observations[indices], replay_buffer.actions[indices]

            # use the sampled data to train an agent, i.e., take gradient step
            train_log = actor.update(ob_batch, ac_batch)
            training_logs.append(train_log)

        # Evaluation and Logging
        print('\nBeginning logging procedure...')

        # decide if videos should be rendered/logged at this iteration
        log_video = ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('\nCollecting video rollouts eval')
            eval_video_trajs = utils.sample_n_trajectories(env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            # save videos
            if eval_video_trajs is not None:
                logger.log_trajs_as_videos(eval_video_trajs, itr, MAX_NVIDEO, fps, 'eval_rollouts')

        # decide if metrics should be logged
        log_metrics = (itr % params['scalar_log_freq'] == 0)
        if log_metrics:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len']
            )
            logs = utils.compute_metrics(trajs, eval_trajs)

            # compute additional metrics
            logs.update(training_logs[-1])  # only use the last log for now
            logs['Train_EnvstepsSoFar'] = total_envsteps
            logs['TimeSinceStart'] = time.time() - start_time

            if itr == 0:
                logs['Initial_DataCollection_AverageReturn'] = logs['Train_AverageReturn']

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            logger.flush()

        if params['save_params']:
            print('\nSaving agent params')
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # experiment id
    parser.add_argument('--env_name', '-env', type=str, help=f"choices: {', '.join(MJ_ENV_NAMES)}", required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--n_iter', '-n', type=int, default=1,
                        help='1 for Vanila BC, otherwise num of DAgger iteration (epoch)')

    # data collection
    parser.add_argument('--ep_len', type=int, help='limit of trajectory sample length')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='training data collected (in the env) during each iteration')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
                        help='validation data collected (in the env) for logging metrics')
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)

    # training
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000,
                        help='number of gradient steps for training policy (per iter in n_iter)')
    parser.add_argument('--train_batch_size', type=int, default=100,
                        help='number of sampled data points to be used per gradient step')

    # policy
    parser.add_argument('--n_layers', type=int, default=2, help='depth, of policy to be learned')
    parser.add_argument('--size', type=int, default=64, help='width of each layer, of policy to be learned')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3, help='lr for supervised learning')

    # logging
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--save_params', action='store_true')

    # seed and GPU
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)

    args = parser.parse_args()

    params = vars(args)  # convert args to dictionary

    # paths and directories
    _cs285 = pathlib.Path(__file__).resolve().parents[1]

    expert_policy_file = str(_cs285 / 'policies' / 'experts' / params['env_name'].split('-')[0]) + '.pkl'
    params['expert_policy_file'] = expert_policy_file

    expert_data = str(_cs285 / 'expert_data' / f"expert_data_{params['env_name']}.pkl")
    params['expert_data'] = expert_data

    _logdir = _cs285.parent / 'run_logs'
    _logdir.mkdir(exist_ok=True)

    if args.do_dagger:
        logdir_prefix = 'q2_'
        assert args.n_iter > 1, ('DAGGER needs more than 1 iteration (n_iter > 1) of training,'
                                 'to iteratively query the expert and train (after 1st warmstarting from BC).')
    else:
        logdir_prefix = 'q1_'
        assert args.n_iter == 1, 'Vanilla BC collects expert data just once (n_iter = 1).'
    logdir = (logdir_prefix + params['exp_name'] + '_' + params['env_name'] + '_' + time.strftime('%d-%m-%Y_%H-%M-%S'))
    logdir = _logdir / logdir
    logdir.mkdir(exist_ok=True)
    params['logdir'] = str(logdir)

    # GO
    run_training_loop(params)


if __name__ == "__main__":
    main()