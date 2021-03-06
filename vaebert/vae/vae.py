import math

from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d_same_padding(
    in_channels: int,
    out_channels: int,
    input_size: Union[int, Sequence[int]],
    kernel_size: Union[int, Sequence[int]],
    stride: Optional[Union[int, Sequence[int]]] = 1,
) -> List[nn.Module]:
    if isinstance(input_size, int):
        input_size = (input_size,) * 3

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3

    if isinstance(stride, int):
        stride = (stride,) * 3

    def _padding_calculation(axis: int) -> int:
        input_axis_size = input_size[axis]
        kernel_axis_size = kernel_size[axis]
        stride_axis_size = stride[axis]
        out_axis_size = (input_axis_size + stride_axis_size - 1) // stride_axis_size
        padding_axis_size = max(
            0,
            (out_axis_size - 1) * stride_axis_size
            + (kernel_axis_size - 1)
            + 1
            - input_axis_size,
        )
        return padding_axis_size

    # Rows is the depth dimension (dim 0), columns is the height dimension (dim 1), and depth is the width dimension (dim 2)
    padding = tuple(map(lambda x: x // 2, (_padding_calculation(i) for i in range(3))))

    # Contains the padding layer (if necessary) and the Conv3d layer
    layers = []

    # Checking if any padding is not even
    padding_odd = list(map(lambda padding_: int(padding_ % 2 != 0), padding))

    if any(padding_odd):
        input_padding = (0, padding_odd[2], 0, padding_odd[1], 0, padding_odd[0])
        layers.append(nn.ConstantPad3d(input_padding, 0))

    layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding))

    return layers


def convtranspose3d_same_padding(
    in_channels: int,
    out_channels: int,
    input_size: Union[int, Sequence[int]],
    kernel_size: Union[int, Sequence[int]],
    stride: Optional[Union[int, Sequence[int]]] = 1,
) -> List[nn.Module]:
    if isinstance(input_size, int):
        input_size = (input_size,) * 3

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3

    if isinstance(stride, int):
        stride = (stride,) * 3

    def _padding_calculation(axis: int) -> float:
        input_axis_size = input_size[axis]
        kernel_axis_size = kernel_size[axis]
        stride_axis_size = stride[axis]
        output_axis_size = (input_axis_size - 1) * stride_axis_size + kernel_axis_size
        padding_axis_size = (
            output_axis_size - (stride_axis_size * input_axis_size)
        ) / 2
        return padding_axis_size

    padding_raw = tuple(_padding_calculation(i) for i in range(3))

    layers = []

    padding_float = list(
        map(lambda padding_: int(not padding_.is_integer()), padding_raw)
    )

    padding = tuple(map(math.ceil, padding_raw))

    if any(padding_float):
        input_padding = (0, padding_float[2], 0, padding_float[1], 0, padding_float[0])
        layers.append(nn.ConstantPad3d(input_padding, 0))

    layers.append(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    )

    return layers


class VAE(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        gen_layers = 5
        self.gen_init_size = input_dim // (2 ** (gen_layers - 1))

        self.reshape_channels = 20
        dropout_rate = 0.15

        self.enc_model = nn.Sequential(
            *conv3d_same_padding(1, 16, input_dim, 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *conv3d_same_padding(16, 32, input_dim // 2, 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *conv3d_same_padding(32, 64, input_dim // (2**2), 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *conv3d_same_padding(64, 128, input_dim // (2**3), 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *conv3d_same_padding(128, 256, input_dim // (2**4), 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * ((self.gen_init_size // 2) ** 3), 2 * self.latent_dim)
        )

        self.latent_gen = nn.Linear(
            self.latent_dim, (self.gen_init_size**3) * self.reshape_channels
        )

        self.gen_model = nn.Sequential(
            *convtranspose3d_same_padding(
                self.reshape_channels, 256, self.gen_init_size, 4, 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *convtranspose3d_same_padding(256, 128, self.gen_init_size * 2, 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *convtranspose3d_same_padding(128, 64, self.gen_init_size * (2**2), 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *convtranspose3d_same_padding(64, 32, self.gen_init_size * (2**3), 4, 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *convtranspose3d_same_padding(32, 1, self.gen_init_size * (2**4), 4, 1)
        )

        self.N = torch.distributions.Normal(0, 1)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logits = self.decode(z)
        probs = torch.sigmoid(x_logits)
        return probs

    def sample(self, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = self.N.sample((5, self.latent_dim))

        return self.decode(eps, apply_sigmoid=True)

    def encode(
        self, x: torch.Tensor, reparam: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.enc_model(x).chunk(2, dim=-1)

        if reparam:
            return self.reparameterize(mean, logvar)

        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = self.N.sample(mean.shape).to(mean.device)
        return eps * torch.exp(logvar * 0.5) + mean

    def decode(
        self, z: torch.Tensor, apply_sigmoid: Optional[bool] = False
    ) -> torch.Tensor:
        z = self.latent_gen(z)
        z = z.reshape(-1, self.reshape_channels, *((self.gen_init_size,) * 3))
        logits = self.gen_model(z)

        if apply_sigmoid:
            probs = torch.sigmoid(logits)
            return probs

        return logits

    def log_normal_pdf(
        self,
        sample: torch.Tensor,
        mean: Union[Number, torch.Tensor],
        logvar: Union[Number, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(logvar, torch.Tensor):
            logvar = torch.tensor(logvar)

        log2pi = torch.log(torch.tensor(2.0 * torch.pi))
        return (
            -0.5 * ((sample - mean) ** 2.0 * torch.exp(-logvar) + logvar + log2pi)
        ).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = F.binary_cross_entropy_with_logits(
            input=x_logit, target=x, reduction="none"
        )
        logpx_z = -cross_ent.sum(dim=(1, 2, 3, 4))
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        loss = (logqz_x - logpz - logpx_z).mean()
        return loss, z
