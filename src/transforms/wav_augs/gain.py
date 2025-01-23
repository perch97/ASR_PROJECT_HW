import torch_audiomentations
from torch import Tensor, nn

class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gain_transform = torch_audiomentations.Gain(*args, **kwargs)

    def forward(self, audio: Tensor):
        expanded_audio = audio.unsqueeze(dim=1)
        processed_audio = self.gain_transform(expanded_audio)
        return processed_audio.squeeze(dim=1)

