"""
Runs behavior cloning and DAgger for homework 1
"""


import pickle
import numpy as np

import torch
import gym

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

    ### ENV ###
    # make the gym environment
    env = gym.make(params['env_name'], render_mode=None)    # make and init the gym env
    _, _ = env.reset(seed=seed)                             # reset the env to an init ob and returns the init ob, info

    # max length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    # env attributes
    assert isinstance(env.action_space, gym.spaces.Box), ("Action space must be continous,"
                                                          "i.e., policy is contionous (conditoonal) distribution.")
    ob_dim = env.observation_space.shape[0]     # condition random variable의 차원
    ac_dim = env.action_space.shape[0]          # random variable의 차원

    # simulation timestep (video 저장에 사용)
    if hasattr(env, 'sim') and hasattr(env.sim, 'model'):   # simulation이 가능한 MuJoCo env라면
        fps = 1 / env.sim.model.opt.timestep
    else:
        fps = env.metadata['render_fps']    # MJ_ENV_NAMES의 env들은 wrapper env가 아닌 base env이므로 env.env 불필요

    ### AGENT (POLICY) ###
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['hidden_size'],
        lr=params['lr']
    )

    # replay buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    ### EXPERT POLICY ###
    _cs285 = Path(__file__).resolve().parents[1]
    _expert_policy = params['env_name'].split('-')[0]
    expert_policy_file = str(_cs285 / 'policies' / 'experts' / _expert_policy)

    print('Loading expert policy from...', expert_policy_file)
    expert_policy = LoadedGaussianPolicy(expert_policy_file)
    expert_policy.to(ptu.device)
    print('Done restoring expert policy.')

    ### TRAINING LOOP ###
    # init vars at beginning of training
    total_envsteps = 0
    start_time = time.time()

    for itr in range(params['n_iter']):
        print(f'\n\n********** Iteration %{itr} ************')

        ## Data Collection (Rolling Out)
        print('\nCollecting data to be used for training...')
        # BC training from expert data
        if itr == 0:
            expert_data = str(_cs285 / 'expert_data' / f"expert_data_{params['env_name']}")
            trajs = pickle.load(open(expert_data, 'rb'))
            envsteps_this_batch = 0
        # DAgger training from sampled data relabeled by expert
        else:
            assert params['do_dagger']
            # sampling transitions
            trajs, envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['batch_size'], params['ep_len']
            )

            print('\nRelabelling collected observations (from policy) with labels from expert policy...')
            for i in range(len(trajs)):
                trajs[i]['acs'] = expert_policy.get_action(trajs[i]['obs'])
                # ---
                # batch의 transition는 get_action()로 모은다.
                # get_action()은 single observation을 받아 single action을 뱉어야 sequential한 sampling이 가능하다.
                # 그러나 여기선 temporal context를 무시하고 batch를 한 번에 expert policy network에 forward passing시켜도 된다.
                # 왜냐하면 DAgger에서 필요한 건,
                # 실제로 expert policy를 env에 굴려 action을 sampling하는 것이 아니라 -- 애초에 expert policy를 env에 굴리는 순간
                # actor의 observation을 공유할 수 없게 된다 --,
                # actor가 모은 (batch) observation의 input에 mapping되는 action output을 얻는 것이기 때문이다.
                # Actor가 모은 batch observation 이미지를 하나씩 보고 human demonstrator가 label을 하나씩 달아 줄 뿐,
                # batch observation의 temporal context는 labeling에 고려하지 않는다.
                # ---
        total_envsteps += envsteps_this_batch
        replay_buffer.add_rollouts(trajs)

        ## Training Agent
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):
            # sample transitions from replay buffer (imitation learning은 observation과 action만 필요)
            indices = np.random.permutation(len(replay_buffer.observations))[:params['train_batch_size']]
            ob_batch, ac_batch = replay_buffer.observations[indices], replay_buffer.actions[indices]

            # take gradient step and log training performance
            train_log = actor.update(ob_batch, ac_batch)
            training_logs.append(train_log)

        ## Evaluate and Log
        print('\nBeginning logging procedure...')
        log_video = ((params['video_log_freq'] != -1) and
                     (itr % params['video_log_freq'] == 0))     # 요번 iter에서 video를 rendering하고 logging할 것인지
        log_metrics = (itr % params['scalar_log_freq'] == 0)    # 요번 iter에서 eval metrics를 logging할 것인지

        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('\nCollecting video of evaluation rollouts...')
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
            print('\nCollecting eval rollouts...')
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len']
            )
            logs = utils.compute_metrics(trajs, eval_trajs)

            # compute additional metrics
            logs.update(training_logs[-1])      # 최신의 training loss 추가
            logs['Train_EnvstepsSoFar'] = total_envsteps
            logs['TimeSinceStart'] = time.time() - start_time
            if itr == 0: logs['Initial_DataCollection_AverageReturn'] = logs['Train_AverageReturn']

            # perform the logging
            for key, value in logs.items():
                print('{}: {}'.format(key, value))
                logger.log_scalar(value, key, iter)
            print('Done logging. \n\n')

            logger.flush()

        if params['save_params']:
            print('\n Saving agent params.')
            filepath = str(Path(params['logdir']) / f'policy_itr_{itr}')
            actor.save(filepath)


if __name__ == "__main__":
    params = parse_args()
    run_training_loop(params)
