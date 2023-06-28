from typing import List, Any
from abc import ABC, abstractmethod
import random
import math
from scipy.signal import butter, filtfilt
from typing import Tuple
import warnings

import numpy as np

from diplom.transformations.base import IAMOnLineTransformation, DatasetRecord


warnings.simplefilter('ignore', np.RankWarning)


class ProbablyApply(IAMOnLineTransformation, ABC):
    def __init__(self, apply_probability: float) -> None:
        assert 0 <= apply_probability <= 1
        self._apply_probability = apply_probability
    
    def __call__(self, item: DatasetRecord) -> None:
        random_number = random.random()
        assert 0 <= random_number <= 1
        if random_number <= self._apply_probability:
            self._call_implementation(item=item)

    @abstractmethod
    def _call_implementation(self, item: DatasetRecord) -> None:
        raise NotImplementedError


class ItalicityAngle(ProbablyApply):
    def __init__(
        self,
        apply_probability: float,
        min_angle_degrees: float,
        max_angle_degrees: float
    ) -> None:
        super().__init__(apply_probability=apply_probability)
        self._min_angle_degrees = min_angle_degrees
        self._max_angle_degrees = max_angle_degrees

    def _call_implementation(self, item: DatasetRecord) -> None:
        features = item["features"]

        xs = features[:, 0]
        ys = features[:, 1]

        alpha_degrees = random.uniform(a=self._min_angle_degrees, b=self._max_angle_degrees)
        alpha_radians = math.radians(alpha_degrees)
        
        x_min = np.min(xs)
        y_min = np.min(ys)

        xp = ((xs - x_min) - ((ys - x_min) * np.tan(alpha_radians))) + x_min
        yp = (ys - y_min) / np.cos(alpha_radians) + y_min

        assert xs.shape == ys.shape

        features[:, 0] = xp
        features[:, 1] = yp

        item["meta"]["alpha"] = alpha_degrees


class BaselineInclinationAngle(ProbablyApply):
    def __init__(
        self,
        apply_probability: float,
        min_angle_radians: float,
        max_angle_radians: float
    ) -> None:
        super().__init__(apply_probability=apply_probability)
        self._min_angle_radians = min_angle_radians
        self._max_angle_radians = max_angle_radians

    def _call_implementation(self, item: DatasetRecord) -> None:
        features = item["features"]

        xs = features[:, 0]
        ys = features[:, 1]

        assert xs.shape == ys.shape

        theta_radians = random.uniform(a=self._min_angle_radians, b=self._max_angle_radians)
        
        xb = xs / np.cos(theta_radians) + ((ys - (xs * np.tan(theta_radians))) * np.sin(theta_radians))
        yb = (ys * np.cos(theta_radians)) - (xs * np.sin(theta_radians))

        features[:, 0] = xb
        features[:, 1] = yb

        item["meta"]["theta"] = theta_radians * 180.0 / 3.141

class ChangeMagnitudeRatio(ProbablyApply):
    def __init__(
        self, 
        apply_probability: float,
        min_coef: float,
        max_coef: float,
    ) -> None:
        super().__init__(apply_probability=apply_probability)
        
        self.min_coef = min_coef
        self.max_coef = max_coef
    

    def _call_implementation(self, item: DatasetRecord) -> None:
        features = item["features"]
        xs = features[:, 0]

        coef = random.uniform(a=self.min_coef, b=self.max_coef)
        xs *= coef
                
        features[:, 0] = xs
        item["meta"]["coef"] = coef


class ChangeFrequency(ProbablyApply):
    def __init__(
        self, 
        apply_probability: float,
        min_coef: float,
        max_coef: float,
    ) -> None:
        super().__init__(apply_probability=apply_probability)
        
        self.min_coef = min_coef
        self.max_coef = max_coef
    

    def _call_implementation(self, item: DatasetRecord) -> None:
        features = item["features"]
        xs = features[:, 0]
        ys = features[:, 1]

        order = 2

        b, a = [], []

        coef = random.uniform(a=self.min_coef, b=self.max_coef)

        normalized_cutoff_freqs = [coef]

        for normalized_cutoff_freq in normalized_cutoff_freqs:
    
            if 0 < normalized_cutoff_freq < 1:
                b_i, a_i = butter(order, normalized_cutoff_freq, btype='low', analog=False)
                b.append(b_i)
                a.append(a_i)

        ys_filtered = []
        xs_filtered = []
        for i in range(len(b)):
            ys_filtered_i = filtfilt(b[i], a[i], ys)
            xs_filtered_i = filtfilt(b[i], a[i], xs)
            ys_filtered.append(ys_filtered_i)
            xs_filtered.append(xs_filtered_i)
                
        assert len(xs_filtered) == 1
        assert features[:, 0].shape == xs_filtered[0].shape
        assert features[:, 1].shape == ys_filtered[0].shape

        features[:, 0] = xs_filtered[0]
        features[:, 1] = ys_filtered[0]
        item["meta"]["freq"] = coef




class BetaElliptic(ProbablyApply):
    def __init__(
        self, 
        apply_probability: float,
        min_x_width_scale: float,
        max_x_width_scale: float,
        y_noise_coef_global_ys_std: float,
        polynom_degree: int,
    ) -> None:
        super().__init__(apply_probability=apply_probability)

        self.min_x_width_scale = min_x_width_scale
        self.max_x_width_scale = max_x_width_scale
        self.polynom_degree = polynom_degree
        self.y_noise_coef_global_ys_std = y_noise_coef_global_ys_std

    @staticmethod
    def get_monotone_intervals_ixes(arr: np.ndarray, offset: int = 0):
        split_points = np.where(np.diff(arr) > 0)[0] + 1
        sub_lists = np.split(np.arange(len(arr)), split_points)
        for sub in sub_lists:
            if sub.size > 1:
                yield sub[0] + offset, sub[-1] + 1 + offset

    def get_intervals_monotone_by_x_from_features(self, features: np.ndarray):
        for stroke_ix0, stroke_ix1 in self._get_strokes_ixes(features):
            xs = features[stroke_ix0:stroke_ix1, 0]
            yield from self.get_monotone_intervals_ixes(xs, offset=stroke_ix0)

    def _poly_aug(self, xs: np.ndarray, ys: np.ndarray, ys_global_std: float) -> Tuple[np.ndarray, np.ndarray]:
        noisy_xs = xs.mean() + (xs - xs.mean()) * np.random.uniform(
            low=self.min_x_width_scale,
            high=self.max_x_width_scale
        )
        noisy_ys = ys + np.random.choice([-1, 1]) * np.random.normal(
            loc=0,
            scale=self.y_noise_coef_global_ys_std * ys_global_std,
            size=len(ys)
        )

        noisy_poly_params = np.polyfit(noisy_xs, noisy_ys, self.polynom_degree)
        polynom = np.poly1d(noisy_poly_params)
        
        new_ys = polynom(xs)
        new_ys[0] = ys[0]
        new_ys[-1] = ys[-1]
        
        return xs, new_ys

    def _call_implementation(self, item: DatasetRecord) -> None:
        features = item["features"]

        ys_std = np.std(features[:, 1])

        for ix0, ix1 in self.get_intervals_monotone_by_x_from_features(features):
            xs = features[ix0:ix1, 0]
            ys = features[ix0:ix1, 1]

            new_xs, new_ys = self._poly_aug(xs=xs, ys=ys, ys_global_std=ys_std)

            features[ix0:ix1, 0] = new_xs
            features[ix0:ix1, 1] = new_ys
