from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from cs285.infrastructure.utils import Trajs


class Logger:
    def __init__(self, log_dir: str | Path, n_logged_samples=10):
        self._log_dir = Path(log_dir)
        print(f'----------------------------------------\n'
              f'Logging outputs to {log_dir}\n'
              f'----------------------------------------')
        self.n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def dump_scalars(self, log_path: str | Path = None):
        log_path = str(self._log_dir / 'scalar_data.json') if log_path is None else str(log_path)
        self._summ_writer.export_scalars_to_json(log_path)

    def log_figure(self, figure: plt.figure, name, step, phase):
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figures(self, figure, name, step, phase):
        assert figure.shape[0] > 0, 'Need [batch_size,n_figures] input for figures logging.'
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_image(self, image, name, step):
        assert (len(image.shape) == 3), 'Need [C,H,W] input for image logging.'
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, 'Need [N,T,C,H,W] input for video logging.'
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_trajs_as_videos(self, trajs: Trajs, step, max_videos_to_save=2, fps=10, video_title='video'):
        # reshape the rollouts ([T,H,W,C] -> [T,C,H,W])
        videos = [np.transpose(traj['image_observation'], [0, 3, 1, 2]) for traj in trajs]

        # get max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length (tensorboard는 frame 수가 동일한 videos만 한번에 logging 가능)
        for i in range(max_videos_to_save):
            # max_length보다 짧은 rollout은 마지막 frame을 max_length까지 반복
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to tensorboard event file
        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def flush(self):
        self._summ_writer.flush()