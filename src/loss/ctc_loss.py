import torch
from torch import Tensor
from torch.nn import CTCLoss
from torch.nn.functional import ctc_loss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)
        loss = ctc_loss(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
            zero_infinity=True,
        )

        return {"loss": loss}
