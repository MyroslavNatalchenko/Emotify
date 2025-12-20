import torch
import torch.nn as nn


class AudioCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_init = nn.BatchNorm2d(1)

        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((4, 4))

        self.elu = nn.ELU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn_init(x)

        x = self.mp_1(self.elu(self.bn_1(self.conv_1(x))))
        x = self.mp_2(self.elu(self.bn_2(self.conv_2(x))))
        x = self.mp_3(self.elu(self.bn_3(self.conv_3(x))))
        x = self.mp_4(self.elu(self.bn_4(self.conv_4(x))))
        x = self.mp_5(self.elu(self.bn_5(self.conv_5(x))))

        x = x.view(x.size(0), -1)
        return x


class MultimodalAudioModel(nn.Module):
    def __init__(self, num_classes=15, mert_input_size=768):
        super().__init__()

        self.cnn_branch = AudioCNNEncoder()

        self.mert_branch = nn.Sequential(
            nn.Linear(mert_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, spec, mert):
        cnn_features = self.cnn_branch(spec)
        mert_features = self.mert_branch(mert)

        combined = torch.cat((cnn_features, mert_features), dim=1)

        logits = self.fusion_layer(combined)

        return logits