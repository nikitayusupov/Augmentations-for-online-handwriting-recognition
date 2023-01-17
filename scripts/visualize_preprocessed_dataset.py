import pickle
import random
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

DATASET_ROOT = Path("data/iam-on-line")
PREPROCESSED_FOLDER = DATASET_ROOT / "preprocessed"
VISUALIZATION_FOLDER = DATASET_ROOT / "preprocessed_visualization"


def visualize_single(lines: List[str], features: np.ndarray, example_ix: int) -> None:
    new_line_features = features[:, -1]
    new_line_ixes = list(np.argwhere(new_line_features).flatten())
    assert len(lines) == len(new_line_ixes)
    new_line_ixes += [len(new_line_features), ]

    plt.figure(figsize=(30, 30))

    for subplot_ix, (ix0, ix1) in enumerate(zip(new_line_ixes[:-1], new_line_ixes[1:])):
        plt.subplot(len(new_line_ixes) - 1, 1, subplot_ix + 1)
        plt.title(lines[subplot_ix], fontsize=25)
        plt.xticks([])
        plt.yticks([])

        is_stroke_start = features[ix0:ix1, 3]
        stroke_start_ixes = list(np.argwhere(is_stroke_start).flatten()) + [len(is_stroke_start), ]

        x = features[ix0:ix1, 0]
        y = features[ix0:ix1, 1]

        for stroke_ix0, stroke_ix1 in zip(stroke_start_ixes[:-1], stroke_start_ixes[1:]):
            plt.plot(x[stroke_ix0:stroke_ix1], -y[stroke_ix0:stroke_ix1], c="blue")

    VISUALIZATION_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(VISUALIZATION_FOLDER / f"{example_ix}.jpg"))
    plt.close()


def visualize_preprocessed_dataset():
    with open(PREPROCESSED_FOLDER / "texts.pickle", "rb") as f:
        texts: List[List[str]] = pickle.load(f)

    with open(PREPROCESSED_FOLDER / "features.pickle", "rb") as f:
        features: List[np.ndarray] = pickle.load(f)

    assert len(texts) == len(features)

    for example_ix, dataset_ix in tqdm(enumerate(random.sample(range(len(texts)), k=50))):
        visualize_single(lines=texts[dataset_ix], features=features[dataset_ix], example_ix=example_ix)


if __name__ == "__main__":
    visualize_preprocessed_dataset()
