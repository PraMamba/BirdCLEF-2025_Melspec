import torch

def compute_deltas(
    specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate"
) -> torch.Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(
        specgram.shape[1], 1, 1
    )

    output = (
        torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    )

    # unpack batch
    output = output.reshape(shape)

    return output


def make_delta(input_tensor: torch.Tensor):
    input_tensor = input_tensor.transpose(3, 2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3, 2)
    return input_tensor


def image_delta(x):
    delta_1 = make_delta(x)
    delta_2 = make_delta(delta_1)
    x = torch.cat([x, delta_1, delta_2], dim=1)
    return x
