from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from tqdm.auto import tqdm
import re
import numpy as np
from copy import deepcopy
import scipy
import pickle
import xml.etree.ElementTree as ET


DATASET_ROOT = Path("data/iam-on-line")
ASCII_FOLDER = DATASET_ROOT / "ascii"
STROKES_FOLDER = DATASET_ROOT / "lineStrokes"
PREPROCESSED_FOLDER = DATASET_ROOT / "preprocessed"

CSR_RE = re.compile(r"^[\s\S]*CSR\:\s+([\s\S]+)$")


@dataclass
class Stroke:
    colour: str
    start_time: float
    end_time: float

    xs: np.ndarray
    ys: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        assert self.xs.ndim == self.ys.ndim == self.times.ndim == 1
        assert self.xs.shape == self.ys.shape == self.times.shape

    def __len__(self) -> int:
        return len(self.xs)


@dataclass
class WhiteboardCaptureSession:
    diagonally_opposite_x: Optional[float]
    diagonally_opposite_y: Optional[float]
    strokes: List[Stroke]


def parse_strokes_from_xml(xml_path: Path) -> WhiteboardCaptureSession:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    diagonally_opposite_x = int(root.find("WhiteboardDescription/DiagonallyOppositeCoords").attrib["x"])
    diagonally_opposite_y = int(root.find("WhiteboardDescription/DiagonallyOppositeCoords").attrib["y"])

    session = WhiteboardCaptureSession(
        diagonally_opposite_y=diagonally_opposite_y,
        diagonally_opposite_x=diagonally_opposite_x,
        strokes=list(),
    )

    for stroke_xml in root.findall("StrokeSet/Stroke"):
        xs_list = []
        ys_list = []
        times_list = []

        for point_xml in stroke_xml.findall("Point"):
            xs_list.append(float(point_xml.attrib["x"]))
            ys_list.append(float(point_xml.attrib["y"]))
            times_list.append(float(point_xml.attrib["time"]))

        colour = stroke_xml.attrib["colour"]
        start_time = float(stroke_xml.attrib["start_time"])
        end_time = float(stroke_xml.attrib["end_time"])

        session.strokes.append(Stroke(
            colour=colour,
            start_time=start_time,
            end_time=end_time,
            xs=np.asarray(xs_list),
            ys=np.asarray(ys_list),
            times=np.asarray(times_list),
        ))

    return session


def extract_features(session: WhiteboardCaptureSession) -> np.ndarray:
    session = deepcopy(session)

    # concatenate xs and ys from all strokes
    xs = np.concatenate([stroke.xs for stroke in session.strokes])
    ys = np.concatenate([stroke.ys for stroke in session.strokes])
    times = np.concatenate([stroke.times for stroke in session.strokes])

    # # calculate mins and maxs
    # assert session.diagonally_opposite_y is not None
    # y_max = session.diagonally_opposite_y

    # x_min = np.min(xs)
    # y_min = np.min(ys)

    # # shift xs such that x[0] = 0
    # xs = xs - x_min

    # # normalize xs and ys so that ys in range [0...1]
    # scale = 1 / (y_max - y_min)
    # xs = (xs - x_min) * scale
    # ys = (ys - y_min) * scale

    # resampling
    # points = np.stack([times, xs, ys], axis=1)
    # resampled_points = []

    feature_4_list: List[float] = list()

    ix0 = 0
    for stroke in session.strokes:
        # num_points = len(stroke)
        # ix1 = ix0 + num_points
        # current_points = points[ix0:ix1]
        # current_points = np.unique(current_points, axis=0)

        # if num_points < 2:
        #     current_resampled_points = current_points
        # else:
        #     diffs = current_points[1:] - current_points[:-1]

        #     # [0, 1] means take into consideration only x,y but not t
        #     distances = np.linalg.norm(diffs, axis=1)
        #     number_of_points = max(2, int(sum(np.linalg.norm(diffs[:, [1, 2]], axis=1)) / 0.05))
        #     u = np.cumsum(distances)
        #     u = np.hstack([[0], u])
        #     t = np.linspace(0, u[-1], number_of_points)
        #     current_resampled_points = scipy.interpolate.interpn((u,), current_points, t)

        # assert len(current_resampled_points) > 0
        # resampled_points.append(current_resampled_points)
        feature_4_list.append(1.0)
        feature_4_list.extend([0.0] * (len(stroke) - 1))

        # ix0 = ix1

    # points = np.concatenate(resampled_points, axis=0)
    # xs = points[:, 1]
    # ys = points[:, 2]
    # times = points[:, 0]

    #
    # feature_0 = np.concatenate(([0, ], xs[1:] - xs[:-1]), dtype=np.float32)
    # feature_1 = np.concatenate(([0, ], ys[1:] - ys[:-1]), dtype=np.float32)
    # feature_2 = np.concatenate(([0, ], times[1:] - times[:-1]), dtype=np.float32)
    # feature_4 = np.asarray(feature_4_list, dtype=np.float32)

    feature_0 = xs
    feature_1 = ys
    feature_2 = times
    feature_4 = np.asarray(feature_4_list, dtype=np.float32)

    assert feature_0.ndim == feature_1.ndim == feature_2.ndim == feature_4.ndim == 1
    assert feature_0.shape == feature_1.shape == feature_2.shape == feature_4.shape

    features = np.stack([
        feature_0,
        feature_1,
        feature_2,
        feature_4,
    ], axis=-1)

    return features


@dataclass
class AnnotatedText:
    """
    data/iam-on-line/ascii/a01/a01-000/a01-000u.txt
    data/iam-on-line/lineStrokes/a01/a01-000/a01-000u-01.xml
    data/iam-on-line/lineStrokes/a01/a01-000/a01-000u-02.xml
    """
    name: str
    txt_path: Path
    xml_paths: List[Path]

    def get_csr_text(self) -> List[str]:
        with open(self.txt_path, "rt") as file:
            text = file.read()

        match = CSR_RE.match(text)
        if not match:
            raise ValueError(f"Failed to parse CSR from file '{self.txt_path}'")

        lines = match.group(1).split("\n")
        lines = list(map(lambda line: line.strip(), lines))  # just in case
        while lines and not lines[-1]:
            lines.pop()  # pop empty lines from the end
        assert all(lines)  # check no empty lines
        return lines

    def get_features(self) -> np.ndarray:
        features_list: List[np.ndarray] = []
        new_line_features_list: List[float] = []

        for xml_path in self.xml_paths:
            session = parse_strokes_from_xml(xml_path=xml_path)
            features_list.append(extract_features(session=session))
            new_line_features_list.extend([1.0, ] + [0.0, ] * (len(features_list[-1]) - 1))

        features = np.concatenate(features_list, axis=0)

        new_line_features = np.asarray(new_line_features_list, dtype=features.dtype)
        new_line_features = new_line_features[:, None]

        features = np.concatenate((features, new_line_features), axis=1)
        return features


def parse_dataset():
    for txt_path in tqdm(
        list(ASCII_FOLDER.rglob("**/*.txt")),
        desc="Iterating ascii folder",
    ):
        name = txt_path.stem
    
        xml_paths: List[Path] = list(STROKES_FOLDER.glob(f"*/*/{name}-*.xml"))
        xml_paths.sort(key=lambda path: int(path.stem.split("-")[-1]))

        if not xml_paths:
            print(f"Failed to find xml files for {txt_path}, skipping...")
            continue
        
        annotated_text = AnnotatedText(
            name=name,
            txt_path=txt_path,
            xml_paths=xml_paths,
        )

        yield annotated_text


def preprocess_dataset():
    PREPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    features = []
    texts = []
    metas = []

    cnt_ok = 0
    cnt_not_equal_number_of_lines = 0
    for annotated_text in parse_dataset():
        lines = annotated_text.get_csr_text()
        if not len(lines) == len(annotated_text.xml_paths):
            cnt_not_equal_number_of_lines += 1
            print(
                f"len(lines) [{len(lines)}] != len(annotated_text.xml_paths) [{len(annotated_text.xml_paths)}] for " 
                f"{annotated_text.txt_path}, skipping..."
            )
            continue

        cnt_ok += 1

        texts.append(lines)
        features.append(annotated_text.get_features())
        metas.append({
            "name": annotated_text.name,
        })

    print(f"cnt_ok: {cnt_ok}")
    print(f"cnt_not_equal_number_of_lines: {cnt_not_equal_number_of_lines}")
    assert len(texts) == len(features)

    with open(PREPROCESSED_FOLDER / "texts.pickle", "wb") as f:
        pickle.dump(texts, f)

    with open(PREPROCESSED_FOLDER / "features.pickle", "wb") as f:
        pickle.dump(features, f)
    
    with open(PREPROCESSED_FOLDER / "metas.pickle", "wb") as f:
        pickle.dump(metas, f)


if __name__ == "__main__":
    preprocess_dataset()
