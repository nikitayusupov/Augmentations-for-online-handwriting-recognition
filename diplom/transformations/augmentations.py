from typing import List, Any
from abc import ABC, abstractmethod
import random
import math

import numpy as np
# import scipy

from diplom.transformations.base import IAMOnLineTransformation, DatasetRecord


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
