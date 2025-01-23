import torch
from torch import nn


class BatchNormT(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x, x_length):
        x = self.bn(x.transpose(1, 2))
        return x.transpose(1, 2), x_length


class ConvBlock(nn.Module):
    def __init__(self, n_conv_layers: int = 2, n_features: int = 128):
        super().__init__()
        assert n_conv_layers in [1, 2, 3]
        self.scaling = 2
        self.out_features = 0
        self.conv_block = [
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        ]

        self.out_features = (n_features + 20 * 2 - 41) // 2 + 1

        if n_conv_layers == 2:
            self.conv_block.extend(
                [
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                ]
            )
            self.out_features = (self.out_features + 10 * 2 - 21) // 2 + 1

        if n_conv_layers == 3:
            self.conv_block.extend(
                [
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=96,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm2d(96),
                    nn.Hardtanh(0, 20, inplace=True),
                ]
            )
            self.out_features = (self.out_features + 20 * 2 - 41) // 2 + 1
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x, x_length):
        x = self.conv_block(x)
        return x, x_length // self.scaling


class GRUBlock(nn.Module):
    """
    Class for RNN layer.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=True,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x, x_length):
        x = nn.utils.rnn.pack_padded_sequence(
            x, x_length, batch_first=True, enforce_sorted=False
        )
        x, _ = self.gru(x, None)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.gru.bidirectional:
            x = x[..., : self.gru.hidden_size] + x[..., self.gru.hidden_size :]
        return x, x_length


class DeepSpeech2(nn.Module):
    """
    Deep Speech 2 from http://proceedings.mlr.press/v48/amodei16.pdf
    """

    def __init__(
        self,
        n_features: int,
        n_tokens: int,
        n_conv_layers: int = 2,
        n_rnn_layers: int = 5,
        fc_hidden: int = 512,
        **batch,
    ):
        super().__init__()

        self.conv_block = ConvBlock(n_conv_layers=n_conv_layers, n_features=n_features)

        rnn_input_size = n_features // 2**n_conv_layers
        rnn_input_size *= 32 if n_conv_layers < 3 else 96

        rnn_output_size = fc_hidden * 2  # bidirectional
        self.gru_layers = [
            GRUBlock(input_size=rnn_input_size, hidden_size=rnn_output_size)
        ]

        for _ in range(n_rnn_layers - 1):
            self.gru_layers.extend(
                [
                    BatchNormT(rnn_output_size),
                    GRUBlock(input_size=rnn_output_size, hidden_size=rnn_output_size),
                ]
            )
        self.gru_layers = nn.Sequential(*self.gru_layers)
        self.batch_norm = nn.BatchNorm1d(rnn_output_size)
        self.fc = nn.Linear(rnn_output_size, n_tokens, bias=False)

    def forward(self, spectrogram, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """

        spectrogram_length = batch["spectrogram_length"]
        spectrogram = spectrogram.unsqueeze(dim=1)
        outputs, output_lengths = self.conv_block(spectrogram, spectrogram_length)

        B, C, F, T = outputs.shape
        outputs = outputs.view(B, C * F, T).transpose(1, 2)

        for gru_layer in self.gru_layers:
            outputs, output_lengths = gru_layer(outputs, output_lengths)

        outputs = self.batch_norm(outputs.transpose(1, 2)).transpose(1, 2)

        log_probs = nn.functional.log_softmax(self.fc(outputs), dim=-1)
        return {"log_probs": log_probs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths // 2

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
