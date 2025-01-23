import torchaudio
from torch import Tensor, nn


class Fading(nn.Module):
    def __init__(self, fade_ratio=0.5, *args, **kwargs):
        super().__init__()
        self.fade_ratio = fade_ratio

    def forward(self, audio: Tensor):
        fade_length = int(audio.size(-1) * self.fade_ratio)
        fade_transform = torchaudio.transforms.Fade(fade_in_len=fade_length, fade_out_len=fade_length)
        audio_expanded = audio.unsqueeze(dim=1)
        faded_audio = fade_transform(audio_expanded)
        return faded_audio.squeeze(dim=1)
