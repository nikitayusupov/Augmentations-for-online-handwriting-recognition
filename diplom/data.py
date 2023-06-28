import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from typing import List, Iterable, Dict, Optional
from pathlib import Path
import pickle

from diplom.transformations.base import IAMOnLineTransformation
from diplom.transformations.registry import build_transformation


class StringLabelEncoder:
    def __init__(self):
        self._encoder = preprocessing.LabelEncoder()

    @staticmethod
    def _split_string(string_: str) -> List[str]:
        return list(string_)
    
    def transform(self, string_: str) -> np.ndarray:
        characters = self._split_string(string_)
        return self._encoder.transform(characters) + 1

    def inverse_transform(self, transformed: Iterable[int]) -> str:
        transformed_shifted = list(map(lambda x: x - 1, transformed))
        if transformed_shifted:
            assert min(transformed_shifted) >= 0
        chars = self._encoder.inverse_transform(transformed_shifted)
        string_ = "".join(chars)
        return string_

    def fit(self, strings: List[str]) -> None:
        characters = list("".join(strings))
        # TODO: sorted for order preseevance
        self._encoder.fit(characters)


class IAMOnLineDataset(Dataset):
    SPLITS_FILES = {
        "train": to_absolute_path("data/iam-on-line/trainset.txt"),
        "val1": to_absolute_path("data/iam-on-line/testset_v.txt"),
        "val2": to_absolute_path("data/iam-on-line/testset_t.txt"),
        "test": to_absolute_path("data/iam-on-line/testset_f.txt"),   
    }

    def __init__(
        self,
        features_path: str = to_absolute_path("data/iam-on-line/preprocessed/features.pickle"),
        texts_path: str = to_absolute_path("data/iam-on-line/preprocessed/texts.pickle"),
        metas_path: str = to_absolute_path("data/iam-on-line/preprocessed/metas.pickle"),
        to_lowercase: bool = True,
        line_by_line: bool = True,
        # transformations: Optional[List[IAMOnLineTransformation]] = None,
        augmentations_before: Optional[List[IAMOnLineTransformation]] = None,
        augmentations_after: Optional[List[IAMOnLineTransformation]] = None,
        normalizations: Optional[List[IAMOnLineTransformation]] = None,
    ):
        self.to_lowercase = to_lowercase
        self.line_by_line = line_by_line
        self.texts_path = texts_path
        self.features_path = features_path
        self.metas_path = metas_path
        # self.transformations = transformations
        self.augmentations_before = augmentations_before
        self.augmentations_after = augmentations_after
        self.normalizations = normalizations

        self.texts: List[str] = self._load_texts()
        self.features: List[np.ndarray] = self._load_features()
        self.metas: List[Dict] = self._load_metas()
        assert len(self.texts) == len(self.features) == len(self.metas)

        self.string_encoder = StringLabelEncoder()
        self.string_encoder.fit(strings=self.texts)
        # print(f"Built IAMOnLineDataset with augmentations before: {self.augmentations_before} and normalizations: {self.normalizations}")
    
    def set_use_augmentations(self, use_augmentations: bool) -> None:
        assert use_augmentations in (True, False)
        self.use_augmentations = use_augmentations

    def _load_metas(self) -> List[Dict]:
        with open(self.metas_path, "rb") as file:
            metas_raw: List[Dict] = pickle.load(file)
        
        with open(self.texts_path, "rb") as file:
            texts_raw: List[List[str]] = pickle.load(file)

        metas: List[Dict] = list()

        if self.line_by_line:
            assert len(metas_raw) == len(texts_raw)
            for meta_raw, text_raw in zip(metas_raw, texts_raw):
                metas.extend([meta_raw] * len(text_raw))
        else:
            raise NotImplementedError

        return metas

    def _load_texts(self) -> List[str]:
        with open(self.texts_path, "rb") as file:
            texts_raw: List[List[str]] = pickle.load(file)
        
        texts: List[str]
        if self.line_by_line:
            texts = [
                line
                for lines in texts_raw
                for line in lines
            ]
        else:
            raise NotImplementedError
        
        if self.to_lowercase:
            texts = list(map(lambda text: text.lower(), texts))
        
        return texts
    
    def _load_features(self) -> List[np.ndarray]:
        with open(self.features_path, "rb") as file:
            features_raw: List[np.ndarray] = pickle.load(file)
        
        features: List[np.ndarray] = list()
        if self.line_by_line:
            for features_raw_single in features_raw:
                new_line_features = features_raw_single[:, -1]
                new_line_ixes = list(np.argwhere(new_line_features).flatten())
                new_line_ixes.append(len(new_line_features))

                for (ix0, ix1) in zip(new_line_ixes[:-1], new_line_ixes[1:]):
                    features.append(features_raw_single[ix0:ix1])
        else:
            raise NotImplementedError
        
        return features

    def get_item_without_transformation(self, idx: int):
        item = {
            "text": self.texts[idx],  # str
            "text_encoded": self.string_encoder.transform(self.texts[idx]),  # numpy array [int]
            "features": self.features[idx],  # numpy array float [T, 5]
            "meta": self.metas[idx],  # some dict - на самом деле в нем просто name файла лежит
        }
        item = deepcopy(item)
        return item

    def __getitem__(self, idx: int):
        item = self.get_item_without_transformation(idx)

        if self.use_augmentations:
            if self.augmentations_before is not None:
                for augmentation in self.augmentations_before:
                    augmentation(item)

        if self.normalizations is not None:
            for normalization in self.normalizations:
                normalization(item)

        if self.use_augmentations:
            if self.augmentations_after is not None:
                for augmentation in self.augmentations_after:
                    augmentation(item)

        return item

    def vis_one_item(self, item, use_cumsum=False, row_ix=0, label='item', row_count=None, idx=None, filename=None, y_arrs=None, x_arrs=None):
        print('shate of y_arrs: ', np.array(y_arrs).shape)
        assert filename is None
        assert row_count is None
        filename = f'/home/nioljusupov/diploma_nikita/item.png'
        assert idx is not None
        assert y_arrs is not None
        row_count = len(y_arrs)
        # Это аналог функции _visualize_item, только рисуем в отдельный файл один вариант текста
        features = item["features"]
        lines = [item["text"]]
        assert use_cumsum == False

        new_line_features = features[:, -1]
        new_line_ixes = list(np.argwhere(new_line_features).flatten())
        assert len(lines) == len(new_line_ixes), (lines, new_line_ixes)
        new_line_ixes += [len(new_line_features), ]
        assert len(lines) == 1     
        assert len(new_line_ixes) == 2

        assert row_ix in (0, 1, 2, 3)
        plt.figure(figsize=(25, 15))
        for ys_idx, (cur_y, cur_x) in reversed(list(enumerate(zip(y_arrs, x_arrs)))):
            for subplot_ix, (ix0, ix1) in reversed(list(enumerate(zip(new_line_ixes[:-1], new_line_ixes[1:])))):
                # print('ix0 =', ix0, 'ix1 =', ix1)
                # print('len of ftrs =', len(features))
                plt.subplot(
                    row_count,  # 4 строчки в сабплоте - до аугментации, после и тд
                    1,  # колонок 1 
                    1 + ys_idx,  # это последовательный сверху-вправо-вниз номер квадратика где щас рисуем
                )

                plt.gca().set_aspect("equal", adjustable='box')
                plt.title(
                    f"{lines[subplot_ix]}",
                        # f"{lines[subplot_ix]}\n({label}, ",
                    # f"alpha: {item['meta'].get('alpha', 0):.2f}, "
                    # f"theta: {item['meta'].get('theta', 0):.1f}, "
                    # f"magnitude: {item['meta'].get('coef', 0):.1f})",
                    fontsize=25
                )

                is_stroke_start = features[ix0:ix1, 3]
                stroke_start_ixes = list(np.argwhere(is_stroke_start).flatten()) + [len(is_stroke_start), ]

                x = features[ix0:ix1, 0]
                # y = features[ix0:ix1, 1]

                for stroke_ix0, stroke_ix1 in zip(stroke_start_ixes[:-1], stroke_start_ixes[1:]):
                    plt.plot(
                        cur_x[stroke_ix0:stroke_ix1],
                        cur_y[stroke_ix0:stroke_ix1],
                        c="blue", lw=1)
                    plt.scatter(cur_x[stroke_ix0:stroke_ix1], cur_y[stroke_ix0:stroke_ix1], c="blue", s=2, alpha=0.5)
        
        plt.savefig(filename)

    def _visualize_item(self, item, use_cumsum: bool, row_ix: int, label: str, row_count: int = 4):
        # Визуализируем один айтем 
        features = item["features"]
        lines = [item["text"]]

        new_line_features = features[:, -1]
        new_line_ixes = list(np.argwhere(new_line_features).flatten())
        assert len(lines) == len(new_line_ixes), (lines, new_line_ixes)
        new_line_ixes += [len(new_line_features), ]
        assert len(lines) == 1     
        assert len(new_line_ixes) == 2

        assert row_ix in (0, 1, 2, 3)
        for subplot_ix, (ix0, ix1) in enumerate(zip(new_line_ixes[:-1], new_line_ixes[1:])):
            plt.subplot(
                row_count,  # 4 строчки в сабплоте - до аугментации, после и тд
                1,  # колонок 1 
                1 + row_ix,  # это последовательный сверху-вправо-вниз номер квадратика где щас рисуем
            )

            plt.gca().set_aspect("equal", adjustable='box')
            plt.title(
                f"{lines[subplot_ix]}\n({label}, "
                f"alpha: {item['meta'].get('alpha', 0):.2f}, "
                f"theta: {item['meta'].get('theta', 0):.1f}, "
                f"magnitude: {item['meta'].get('coef', 0):.1f})",
                fontsize=25
            )

            is_stroke_start = features[ix0:ix1, 3]
            stroke_start_ixes = list(np.argwhere(is_stroke_start).flatten()) + [len(is_stroke_start), ]

            x = features[ix0:ix1, 0]
            y = features[ix0:ix1, 1]
            if use_cumsum:
                x = np.cumsum(x)
                y = np.cumsum(y)

            for stroke_ix0, stroke_ix1 in zip(stroke_start_ixes[:-1], stroke_start_ixes[1:]):
                plt.plot(
                    x[stroke_ix0:stroke_ix1],
                    -y[stroke_ix0:stroke_ix1],
                    c="blue", lw=1)
                plt.scatter(x[stroke_ix0:stroke_ix1], -y[stroke_ix0:stroke_ix1], c="blue", s=2, alpha=0.5)

    def visualize(self, idx: int, image_path: str) -> None:
        # Функция рисует и сохраняет в файл картинку -- визуализиаця строчки до транформаций и после
        raw_item = self.get_item_without_transformation(idx) # айтем до трансформаций
        aug_before_item = deepcopy(raw_item)
        if self.augmentations_before is not None:
            for augmentation in self.augmentations_before:
                augmentation(aug_before_item)

        norm_no_aug_item = deepcopy(raw_item)
        if self.normalizations is not None:
            for normalization in self.normalizations:
                normalization(norm_no_aug_item)

        final = deepcopy(aug_before_item)
        if self.normalizations is not None:
            for normalization in self.normalizations:
                normalization(final)
        
        plt.figure(figsize=(25, 15))
        plt.gca().set_aspect("equal", adjustable='box')

        self._visualize_item(raw_item, use_cumsum=False, row_ix=0, label="before any transformations") # до всего
        self._visualize_item(aug_before_item, use_cumsum=False, row_ix=1, label="after only augmentation") # после аугментации1
        self._visualize_item(norm_no_aug_item, use_cumsum=True, row_ix=2, label="after only normalization") # после норм только
        self._visualize_item(final, use_cumsum=True, row_ix=3, label="after normalization & augmentation") # после норм + ауг2
        
        Path(image_path).parent.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
    
    def __len__(self) -> int:
        return len(self.texts)

    def get_split_ixes(self, split_name: str) -> List[int]:
        assert split_name in self.SPLITS_FILES
        
        with open(self.SPLITS_FILES[split_name], "rt") as f:
            target_names: List[str] = f.readlines()
            target_names = list(map(lambda line: line.strip(), target_names))
            target_names = list(filter(lambda line: len(line) > 0, target_names))

        idxes: List[int] = list()
        for ix in range(len(self)):
            name = self.metas[ix]["name"]
            if name in target_names:
                idxes.append(ix)
        
        assert len(set(idxes)) == len(idxes)
        return idxes


def pad(arrays: List[np.ndarray], axis: int):
    # [0 1 2]    [0 1 2 3]

    max_len = 0
    for array in arrays:
        max_len = max(max_len, array.shape[axis])

    arrays_padded = list()
    original_lengths: List[int] = []
    for array in arrays:
        pad_width = [[0, 0] for _ in range(array.ndim)]
        original_lengths.append(array.shape[axis])
        pad_width[axis][1] = max_len - array.shape[axis]
        arrays_padded.append(np.pad(array, pad_width=pad_width))

    assert len(arrays_padded) == len(original_lengths)
    
    return arrays_padded, original_lengths


def i_am_online_collate_fn(batch_elements: List):
    # return {
    #         "text": self.texts[idx],  # str
    #         "text_encoded": self.string_encoder.transform(self.texts[idx]),  # numpy array [int]
    #         "features": self.features[idx],  # numpy array float [T, 5]
    #     }

    texts = [element["text"] for element in batch_elements]

    texts_encoded_padded, texts_lengths = pad(
        arrays=[element["text_encoded"] for element in batch_elements],
        axis=0,
    )

    features_padded, features_lengths = pad(
        arrays=[element["features"] for element in batch_elements],
        axis=0,
    )

    return {
        "texts": texts,
        "texts_encoded_padded": torch.LongTensor(np.array(texts_encoded_padded)),
        "texts_lengths": torch.LongTensor(texts_lengths),
        "features_padded": torch.FloatTensor(np.array(features_padded)),
        "features_lengths": torch.LongTensor(features_lengths),
    }


def build_i_am_online_datasets(dataset_cfg):
    dataset_cfg = deepcopy(dataset_cfg)
    # transformations_cfg = dataset_cfg.pop("transformations")
    augmentations_before_cfg = dataset_cfg.pop("augmentations_before")
    augmentations_after_cfg = dataset_cfg.pop("augmentations_after")
    normalizations_cfg = dataset_cfg.pop("normalizations")

    # transformations = []
    # for transformation_cfg in transformations_cfg:
    #     name = transformation_cfg.pop("name")
    #     transformations.append(build_transformation(
    #         name=name,
    #         **transformation_cfg
    #     ))

    augmentations_before = []
    if augmentations_before_cfg:
        for augmentation_cfg in augmentations_before_cfg:
            name = augmentation_cfg.pop("name")
            augmentations_before.append(build_transformation(
                name=name,
                **augmentation_cfg
            ))

    normalizations = []
    for normalization_cfg in normalizations_cfg:
        name = normalization_cfg.pop("name")
        normalizations.append(build_transformation(
            name=name,
            **normalization_cfg
        ))    

    augmentations_after = []
    if augmentations_after_cfg:
        for augmentation_cfg in augmentations_after_cfg:
            name = augmentation_cfg.pop("name")
            augmentations_after.append(build_transformation(
                name=name,
                **augmentation_cfg
            ))

    # dataset = IAMOnLineDataset(
    #     # transformations=transformations,
    #     augmentations=augmentations,
    #     normalizations=normalizations,
    #     **dataset_cfg
    # )

    dataset_for_train = IAMOnLineDataset(
        augmentations_before=augmentations_before,
        augmentations_after=augmentations_after,
        normalizations=normalizations,
        **dataset_cfg
    )
    dataset_for_train.set_use_augmentations(True)

    dataset_for_test = IAMOnLineDataset(
        augmentations_before=None,
        augmentations_after=None,
        normalizations=normalizations,
        **dataset_cfg
    )
    dataset_for_test.set_use_augmentations(False)

    assert dataset_for_train.get_split_ixes("train") == dataset_for_test.get_split_ixes("train") 
    assert dataset_for_train.get_split_ixes("val1") == dataset_for_test.get_split_ixes("val1") 
    assert dataset_for_train.get_split_ixes("val2") == dataset_for_test.get_split_ixes("val2") 
    assert dataset_for_train.get_split_ixes("test") == dataset_for_test.get_split_ixes("test") 
    train_ixes = dataset_for_train.get_split_ixes("train")
    val1_ixes = dataset_for_train.get_split_ixes("val1")
    val2_ixes = dataset_for_train.get_split_ixes("val2")
    test_ixes = dataset_for_train.get_split_ixes("test")

    assert len(train_ixes) + len(val1_ixes) + len(val2_ixes) + len(test_ixes) \
    == len(set(train_ixes) | set(val1_ixes) | set(val2_ixes) | set(test_ixes))

    # train_dataset = torch.utils.data.Subset(dataset, train_ixes)
    train_dataset = torch.utils.data.Subset(dataset_for_train, train_ixes)
    val_dataset = torch.utils.data.Subset(dataset_for_test, val1_ixes + val2_ixes)
    test_dataset = torch.utils.data.Subset(dataset_for_test, test_ixes)
        
    return {
        "all": dataset_for_train,
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


if __name__ == "__main__":
    dataset = IAMOnLineDataset(to_lowercase=True, line_by_line=True)
    print(dataset)

    le = StringLabelEncoder()
    le.fit(["gay", "shit fuck"])
    print(le.transform("gay"))
    print(le._encoder.classes_)

    print(dataset[0])
    print(len(dataset))

    print(pad(
        arrays=[np.asarray([[0,1,2], [1,2,3]]), np.asarray([[2,2,8],])],
        axis=0,
    ))

    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=i_am_online_collate_fn,
    )

    for batch in loader:
        print(batch.keys())
        print(batch)
        break
