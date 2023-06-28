import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate
import wandb
import pickle
import warnings

from tqdm.auto import tqdm
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from scipy.signal import savgol_filter, butter, filtfilt


from diplom.data import build_i_am_online_datasets, i_am_online_collate_fn
from diplom.model import IAMOnLineModel
from diplom.postprocessing import IAMOnLineCTCDecoder, IAMOnLineCTCDecoderMultiprocessed
from diplom.utils import get_strokes_ixes
from diplom.transformations.augmentations import BetaElliptic


warnings.simplefilter('ignore', np.RankWarning)


def calculate_derivative(f_t, t):
    assert f_t.shape == t.shape
    assert f_t.ndim == t.ndim == 1
    diff_f_t = f_t[1:] - f_t[:-1]
    diff_t = t[1:] - t[:-1]

    one_side_derivative = diff_f_t / (diff_t + 1e-8)

    two_side_derivative = 0.5 * (one_side_derivative[1:] + one_side_derivative[:-1])
    two_side_derivative = np.concatenate((
        (one_side_derivative[0],),
        two_side_derivative,
        (one_side_derivative[-1],),
    ))
    assert two_side_derivative.shape == f_t.shape
    return two_side_derivative


def calculate_stroke_velocity(stroke):
    stroke_x = stroke[:, 0]
    stroke_y = stroke[:, 1]
    stroke_t = stroke[:, 2]
    dxdt = calculate_derivative(stroke_x, stroke_t)
    dydt = calculate_derivative(stroke_y, stroke_t)
    velocity = np.sqrt(dxdt ** 2 + dydt ** 2)
    return velocity


def get_ixes_between_velocity_extremum_points(features):
    from scipy.signal import argrelextrema

    for stroke_ix0, stroke_ix1 in get_strokes_ixes(features[:, 3]):
        assert stroke_ix0 < stroke_ix1

        stroke = features[stroke_ix0:stroke_ix1]
        velocity = calculate_stroke_velocity(stroke)

        ixes = stroke_ix0 + np.concatenate((
            argrelextrema(velocity, np.less)[0] + 1,
            argrelextrema(velocity, np.greater)[0] + 1,
            [0, len(velocity)],
        ))
        ixes = np.sort(ixes)
        for ix0, ix1 in zip(ixes[:-1], ixes[1:]):
            yield ix0, ix1


def get_monotone_intervals_ixes(arr, offset=0):
    split_points = np.where(np.diff(arr) > 0)[0] + 1
    sub_lists = np.split(np.arange(len(arr)), split_points)
    for sub in sub_lists:
        if sub.size > 1:
            yield sub[0] + offset, sub[-1] + 1 + offset


def get_intervals_monotone_by_x_from_features(features):
    print("here")
    for stroke_ix0, stroke_ix1 in get_strokes_ixes(features[:, 3]):
        print("stroke_len",stroke_ix1-stroke_ix0)
        xs = features[stroke_ix0:stroke_ix1, 0]
        yield from get_monotone_intervals_ixes(xs, offset=stroke_ix0)


@hydra.main(config_path="../configs", config_name="no_lm_no_aug_case_sensetive")
def train(cfg: DictConfig) -> None:
    print('Try to build_i_am_online_datasets')
    datasets = build_i_am_online_datasets(OmegaConf.to_container(cfg.dataset))
    print('Did build datasets\n')

    all_data = datasets['all']    
    idx = random.randint(0, len(all_data) - 1)
    from pathlib import Path
    for idx in tqdm(range(600, 650)):
        # idx = 998
        print('idx =', idx)

        item = all_data.get_item_without_transformation(idx)
        
        features = item["features"]

        
        viz_folder = Path("new_aug_viz")
        if viz_folder.is_dir():
            shutil.rmtree(viz_folder)

        aug = BetaElliptic(
            apply_probability=1.0,
            min_x_width_scale=0.75,
            max_x_width_scale=1.25,
            y_noise_coef_global_ys_std=0.2,
            polynom_degree=2,
        )

        plt.figure(figsize=(30, 30))
        for ix0, ix1 in get_strokes_ixes(features[:, 3]):
            plt.subplot(511)
            xs = features[ix0:ix1, 0]
            ys = -features[ix0:ix1, 1]
            plt.plot(xs, ys, c="blue", lw=1)
        
        ys_std = np.std(features[:, 1])

        colors = ("red", "orange", "black", "brown")
        for attempt in range(4):
            plt.subplot(5,1,attempt+2)
            item_augmented = deepcopy(item)
            aug(item_augmented)
            features_augmented = item_augmented["features"]

            for ix0, ix1 in get_strokes_ixes(features_augmented[:, 3]):
                xs = features_augmented[ix0:ix1, 0]
                ys = -features_augmented[ix0:ix1, 1]
                plt.plot(xs, ys, c="blue", lw=1)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(f"/home/nioljusupov/diploma_nikita/vVviz_std0.2/aug_only_poly2_ix_{idx}.jpg")
        # plt.clear()

    # all_data.visualize(1000, "/home/nioljusupov/diploma_nikita/gt_vVviz.jpg")


if __name__ == "__main__":
    train()

