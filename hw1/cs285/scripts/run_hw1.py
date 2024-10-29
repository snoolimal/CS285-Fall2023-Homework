import gymnasium as gym

import torch

import pickle
import time
from tqdm import tqdm

from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_guassian_policy import LoadedGaussianPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.logger import Logger
from cs285.scripts.params import ArgParser


def run_training_loop(params):
    """
    Runs BC or DAgger training with the specified parameters.
    ---
    Args:
        params: specified experiment parameters
    """
    # Basic Setting
    # create logger
    logger = Logger(params['log_dir'])

    # set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set gpu
    ptu.init_gpu(
        not params['no_gpu'],
        params['which_gpu']
    )

    # Environment
    # initialization
    env = gym.make(params['env_name'], render_mode=None)   # make and init the gym env
    _, _ = env.reset(seed=seed)                            # resets the env to an init ob and returns the init ob, info

    assert isinstance(env.action_space, gym.spaces.Box), \
        "Action space must be continous."                               # action space 유형 확인 (continous)

    ob_dim = env.observation_space.shape[0]                             # observation space dimension
    ac_dim = env.action_space.shape[0]                                  # action space dimension
    # annotation: gym.space.Box

    # Log Setting
    scalar_log_freq = params['scalar_log_freq']
    video_log_feq = params['video_log_freq']

    # set simulation tstep (video 저장에 사용)
    if 'model' in dir(env):
        fps = 1 / env.model.opt.timestep        # MuJoCo 기반 gym env는 요 attr로 simul의 각 단계의 길이를 지정
    else:
        fps = env.metadata['render_fps']        # meadata attr에는 env의 기본 정보들이 포함

    # Agent
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['hidden_size'],
        params['learning_rate']
    )

    # Replay Buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    # Expert Policy
    filename = params['expert_pol_path']
    print('Loading expert policy from...', filename)
    expert_policy = LoadedGaussianPolicy(filename).to(ptu.device)
    print('Restoring expert policy is done.\n')

    # Training Loop
    n_iter = params['n_iter']
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps  # episode의 최대 길이

    total_envsteps = 0
    start_time = time.time()

    pbar = tqdm(range(n_iter), total=n_iter, position=0, leave=True)
    for itr in pbar:
        pbar.set_description(desc=f'Iteration {itr}')

        log_video = ((itr % video_log_feq == 0) and (video_log_feq != -1))
        log_metrics = itr % scalar_log_freq == 0

        # BC training from expert data를 위한 trajs 수집 or DAgger를 위한 1st BC iteration 수행
        if itr == 0:
            s = 'First BC iteration for DAgger training...' if params['do_dagger'] else \
                'Collecting trajs to be used for BC training...'
            pbar.set_postfix_str(s=s, refresh=True)

            trajs = pickle.load(open(params['expert_data_path'], 'rb'))
            _ = {'observation': 'obs', 'action': 'acs', 'reward': 'rwds', 'next_observation': 'next_obs', 'terminal': 'terminals'}
            for i in range(len(trajs)):
                trajs[i] = {_.get(k, k): v for k, v in trajs[i].items()}

            envsteps_this_batch = 0
        # DAGGER training from sampled data relabeled by expert을 위한 trajs 수집
        else:
            assert params['do_dagger']
            trajs, envsteps_this_batch = TrajSampler(env, actor, params['ep_len']).batch_mintstep(params['batch_size'])

            pbar.set_postfix_str(s='Relabelling collected observations with labels from an expert policy...',
                                 refresh=True)
            for traj in trajs:
                # new_acs = [expert_policy.get_action(ob) for ob in traj['obs']]   # 굳이 for문을 돌 필요 없이
                new_acs = expert_policy.get_action(traj['obs'])                    # batch 처리
                assert new_acs.shape == traj['acs'].shape
                traj['acs'] = new_acs

        total_envsteps += envsteps_this_batch
        replay_buffer.add_rollouts(trajs)   # (수집한 -- DAgger라면 relabel된 -- rollouts를 buffer에 추가)

        # Train agent (using sampled data from replay buffer)
        training_logs = []

        train_pbar = tqdm(range(params['num_agent_train_steps_per_iter']), position=0, leave=False,
                          desc=f'Iteration {itr} | Training agent using sampled data from replay buffer...')
        for _ in train_pbar:
            indices = np.random.permutation(params['train_batch_size'])
            # for imitation learning, we only need observations and actions
            ob_batch, ac_batch = replay_buffer.obs[indices], replay_buffer.acs[indices]

            train_log = actor.update(ob_batch, ac_batch)
            training_logs.append(train_log)

        pbar.set_postfix_str('Beginning logging procedure...', refresh=True)
        if log_video:
            # save eval rollouts as videos in tensorboard event file
            pbar.set_postfix_str('Collecting video rollouts eval...', refresh=True)
            eval_video_paths = TrajSampler(
                env, actor, params['ep_len'], True
            ).batch_ntraj(params['max_videos_to_save'])

            # save videos
            if eval_video_paths is not None:
                logger.log_trajs_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=params['max_videos_to_save'],
                    video_title='eval_rollouts'
                )

        if log_metrics:
            pbar.set_postfix_str('Collecting data for eval...', refresh=True)
            eval_trajs, _ = TrajSampler(
                env, actor, params['ep_len']
            ).batch_mintstep(params['eval_batch_size'])     # _: eval_envsteps_this_batch

            logs = compute_metrics(trajs, eval_trajs)
            logs.update(training_logs[-1])
            logs['Train_EnvstepsSoFar'] = total_envsteps
            logs['TimeSinceStart'] = time.time() - start_time
            if itr == 0:
                logs['Initial_DataCollection_AverageReturn'] = logs['Train_AverageReturn']

            for key, value in logs.items():
                # print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            pbar.set_postfix_str('Done logging...', refresh=True)

            logger.flush()

            if params['save_params']:
                pbar.set_postfix_str('Saving agent params', refresh=True)
                actor.save('{}/policy_itr_{}.pt'.format(params['log_dir'], itr))


def main(debug=False):
    if debug:
        params = ArgParser.dummy_params_for_debug()
    else:
        params = ArgParser.parse_args()

    run_training_loop(params)


if __name__ == "__main__":
    main(debug=True)
