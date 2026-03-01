import torch
import torch.nn as nn

class PronunciationModel(nn.Module):
    def __init__(self, fluency_input_size, prosody_input_size):
        super(PronunciationModel, self).__init__()

        # self.accuracy_layers = nn.Sequential(
        #     nn.Linear(accuracy_input_size, 32),
        #     nn.ReLU(),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 1)
        # )

        self.fluency_layers = nn.Sequential(
            nn.Linear(fluency_input_size, 32),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.prosody_layers = nn.Sequential(
            nn.Linear(prosody_input_size, 32),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, fluency_features, prosody_features):
        fluency_score = self.fluency_layers(fluency_features)
        prosody_score = self.prosody_layers(prosody_features)

        return fluency_score, prosody_score
