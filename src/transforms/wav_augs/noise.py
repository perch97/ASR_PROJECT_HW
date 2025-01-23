import torch_audiomentations
from torch import Tensor, nn

class AddColoredNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.noise_transform = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def forward(self, audio: Tensor):
        expanded_audio = audio.unsqueeze(dim=1)
        noisy_audio = self.noise_transform(expanded_audio)
        return noisy_audio.squeeze(dim=1)
