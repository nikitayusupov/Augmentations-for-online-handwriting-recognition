import torch
import torch.nn as nn


class IAMOnLineModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        self.lstms = nn.LSTM(
            in_features,
            32,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        features, _ = self.lstms(x)

        logits = self.fc(features)  # to which dim is it applied?
        logprobs = nn.functional.log_softmax(logits, dim=2)
        return logprobs
