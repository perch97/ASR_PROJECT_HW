import torch_audiomentations
from torch import Tensor, nn

class BandPassFilter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.band_pass_transform = torch_audiomentations.BandPassFilter(*args, **kwargs)

    def __call__(self, data: Tensor):
        expanded_audio = data.unsqueeze(dim=1)
        filtered_audio = self.band_pass_transform(expanded_audio)
        return filtered_audio.squeeze(dim=1)


class BandStopFilter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.band_stop_transform = torch_audiomentations.BandStopFilter(*args, **kwargs)

    def __call__(self, data: Tensor):
        expanded_audio = data.unsqueeze(dim=1)
        filtered_audio = self.band_stop_transform(expanded_audio)
        return filtered_audio.squeeze(dim=1)