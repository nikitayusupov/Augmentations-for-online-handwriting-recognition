from diplom.transformations.base import IAMOnLineTransformation
from diplom.transformations.preprocessing import (
    Normalize,
    CalculateDifferencesBetweenAdjacentPoints,
    EquidistantResample
)
from diplom.transformations.augmentations import (
    ItalicityAngle,
)

REGISTRY = {
    "Normalize": Normalize,
    "CalculateDifferencesBetweenAdjacentPoints": CalculateDifferencesBetweenAdjacentPoints,
    "EquidistantResample": EquidistantResample,
    "ItalicityAngle": ItalicityAngle,
}


def build_transformation(name: str, **kwargs) -> IAMOnLineTransformation:
    cls = REGISTRY[name]
    return cls(**kwargs)
