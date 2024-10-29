from pathlib import Path
import matplotlib.pyplot
import numpy as np
from tensorboardX import SummaryWriter


class Logger:
    """Logger Using TensorBoardX"""
    def __init__(self, log_dir):
        self._log_dir = log_dir
        print(f'Logging outputs to {log_dir}')

        # https://keep-steady.tistory.com/14
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """
        그룹으로 묶은 scalar들을 하나의 plot에 logging한다.
        """
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert len(image.shape) == 3, "Image logging requires [C,H,W] input tensor."
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_figure(self, figure: matplotlib.pyplot.figure, name, step, phase):
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figures(self, figures: matplotlib.pyplot.figure, name, step, phase):
        assert figures.shape[0] > 0, "Figures logging requires input shape [batch x figures]."
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figures, step)

    def log_graph(self, array, name, step, phase):
        assert len(array.shape) == 3, "Array of image logging requires input shape [C,H,W]."
        self._summ_writer.add_image('{}_{}'.format(name, phase), array, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Video logging requires [N,T,C,H,W] input tensor."
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_trajs_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):
        """
        TensorBoardX는 영상 기록 시 그 길이 -- i.e., frame 수 -- 가 모두 같아야 한다.
        여기서 각 영상은 agent가 구른 rollout(trajectory)를 보여 준다.
        """
        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]    # [T,H,W,C] -> [T,C,H,W]

        # num of video to save
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])

        # get max rollout length
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all videos be same length
        for i in range(max_videos_to_save):
            # 최대 길이보다 짧은 rollout은 마지막 frame을 최대 길이까지 반복
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to TensorBoard event file
        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def dump_scalars(self, log_path=None):
        """
        TensorBoard에 기록된 모든 scalars를 JSON 파일로 export한다.
        """
        log_path = Path(self._log_dir) / 'scalar_data.json' if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        """
        메모리에 버퍼링된 모든 event를 디스크에 강제로 기록해
        프로그램이 비정상 종료될 때 데이터의 손실을 방지하고
        실시간으로 TensorBoard를 업데이트하며
        소요 시간이 긴 학습의 중간마다 데이터를 확실히 저장한다.
        """
        self._summ_writer.flush()
