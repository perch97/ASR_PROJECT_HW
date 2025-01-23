from torch import Tensor, nn
from torchaudio.transforms import SpeedPerturbation


class SpeedPerturb(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.speed_transform = SpeedPerturbation(*args, **kwargs)

    def __call__(self, data: Tensor):
        expanded_audio = data.unsqueeze(dim=1)
        perturbed_audio = self.speed_transform(expanded_audio)[0]
        return perturbed_audio.squeeze(dim=1)
