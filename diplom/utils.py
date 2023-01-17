from typing import Generator, Tuple

import numpy as np


def get_strokes_ixes(
    is_stroke_start_feature: np.ndarray
) -> Generator[Tuple[int, int], None, None]:
    stroke_start_ixes = list(np.argwhere(is_stroke_start_feature).flatten())
    stroke_start_ixes.append(len(is_stroke_start_feature))

    for stroke_ix0, stroke_ix1 in zip(stroke_start_ixes[:-1], stroke_start_ixes[1:]):
        yield stroke_ix0, stroke_ix1
