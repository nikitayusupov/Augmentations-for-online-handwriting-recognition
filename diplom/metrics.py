from multiprocessing import Pool
from typing import List, Tuple
from operator import itemgetter

import numpy as np
from torchmetrics import CharErrorRate


def get_CER_numerator_denominator(
    predictions: List[str],
    targets: List[str],
) -> Tuple[float, float]:
    cer = CharErrorRate()
    cer.update(preds=predictions, target=targets)
    return float(cer.errors.item()), float(cer.total.item())


def calculate_CER_multiproc(
    predictions: List[str],
    targets: List[str],
    n_jobs: int,
):
    assert len(predictions) == len(targets)
    n_jobs = min(n_jobs, len(predictions))

    chunks = list(map(
        list,
        np.array_split(list(zip(predictions, targets)), n_jobs)
    ))

    with Pool(processes=n_jobs) as pool:
        calculations = pool.map(
            lambda pt: get_CER_numerator_denominator(predictions=pt[0], targets=pt[1]),
            chunks
        )

    numerator = sum(map(itemgetter(0), calculations))
    denominator = sum(map(itemgetter(1), calculations))

    cer_metric = numerator / denominator
    return cer_metric
