from typing import List

import numpy as np
import scipy

from diplom.transformations.base import IAMOnLineTransformation, DatasetRecord


class Normalize(IAMOnLineTransformation):
    def __call__(self, item: DatasetRecord) -> None:
        features = item["features"]

        xs = features[:, 0]
        ys = features[:, 1]

        assert xs.shape == ys.shape

        # calculate mins and maxs
        y_max = np.max(ys)
        x_min = np.min(xs)
        y_min = np.min(ys)

        # normalize xs and ys so that ys in range [0...1]
        scale = 1 / (y_max - y_min)
        xs = (xs - x_min) * scale
        ys = (ys - y_min) * scale

        # TODO: mb increase by 20 % bbox?

        features[:, 0] = xs
        features[:, 1] = ys


class CalculateDifferencesBetweenAdjacentPoints(IAMOnLineTransformation):
    def __call__(self, item: DatasetRecord) -> None:
        features = item["features"]

        xs = features[:, 0]
        ys = features[:, 1]
        times = features[:, 2]

        assert xs.shape == ys.shape

        dx = np.concatenate(([0, ], xs[1:] - xs[:-1]), dtype=np.float32)
        dy = np.concatenate(([0, ], ys[1:] - ys[:-1]), dtype=np.float32)
        dtimes = np.concatenate(([0, ], times[1:] - times[:-1]), dtype=np.float32)

        features[:, 0] = dx
        features[:, 1] = dy
        features[:, 2] = dtimes


class EquidistantResample(IAMOnLineTransformation):
    def __init__(self, delta: float):
        self._delta = delta

    def __call__(self, item: DatasetRecord) -> None:
        features = item["features"]
        assert features.shape[1] == 5  # x, y, t, is_stroke_start, is_line_start
        xs = features[:, 0]
        ys = features[:, 1]
        times = features[:, 2]

        points = np.stack([times, xs, ys], axis=1)

        resampled_points: List[np.ndarray] = list()

        is_stroke_start_list: List[int] = list()
        is_line_start_list: List[int] = list()

        for ix0, ix1 in self._get_strokes_ixes(features):
            num_points = ix1 - ix0

            current_points = points[ix0:ix1]
            current_points = np.unique(current_points, axis=0)

            if num_points < 2:
                current_resampled_points = current_points
            else:
                diffs = current_points[1:] - current_points[:-1]

                # [0, 1] means take into consideration only x,y but not t
                distances = np.linalg.norm(diffs, ord=2, axis=1)
                number_of_points = max(
                    2,
                    int(sum(np.linalg.norm(diffs[:, [1, 2]], axis=1, ord=2)) / self._delta)
                )
                u = np.cumsum(distances)
                u = np.hstack([[0], u])
                t = np.linspace(0, u[-1], number_of_points)
                current_resampled_points = scipy.interpolate.interpn((u,), current_points, t)

            assert len(current_resampled_points) > 0
            resampled_points.append(current_resampled_points)
                
            is_stroke_start_list.append(1.0)
            is_stroke_start_list.extend([
                0 for _ in range(len(current_resampled_points) - 1)
            ])

            is_line_start_list.append(features[ix0, 4])
            is_line_start_list.extend([
                0 for _ in range(len(current_resampled_points) - 1)
            ])

        points = np.concatenate(resampled_points, axis=0)
        xs = points[:, 1]
        ys = points[:, 2]
        times = points[:, 0]

        new_features = np.stack((
            xs,
            ys,
            times,
            np.asarray(is_stroke_start_list, dtype=np.float32),
            np.asarray(is_line_start_list, dtype=np.float32),
        ), axis=1)

        item["features"] = new_features
