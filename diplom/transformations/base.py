from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Tuple

import numpy as np

from diplom.utils import get_strokes_ixes


DatasetRecord = Dict[str, Any]


class IAMOnLineTransformation(ABC):
    @abstractmethod
    def __call__(self, item: DatasetRecord) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_strokes_ixes(features: np.ndarray) -> Generator[Tuple[int, int], None, None]:
        yield from get_strokes_ixes(
            is_stroke_start_feature=features[:, 3],
        )
