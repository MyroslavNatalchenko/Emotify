import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(AudioCNN, self).__init__()

        # Input: (Batch, 1, 96, 1366)

        self.bn_init = nn.BatchNorm2d(1)

        # Layer 1: (1, 96, 1366) -> (64, 48, 341)
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # Layer 2: (64, 48, 341) -> (128, 24, 85)
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # Layer 3: (128, 24, 85) -> (128, 12, 21)
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # Layer 4: (128, 12, 21) -> (128, 4, 4)
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # Layer 5: (128, 4, 4) -> (64, 1, 1)
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((4, 4))

        # Classifier
        self.dense = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # (Batch, 96, 1366) -> (Batch, 1, 96, 1366)
        x = x.unsqueeze(1)

        x = self.bn_init(x)

        # layer 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))
        # layer 2
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))
        # layer 3
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))
        # layer 4
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))
        # layer 5
        x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))

        # Flatten: (Batch, 64, 1, 1) -> (Batch, 64)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        logit = self.dense(x)

        return logit