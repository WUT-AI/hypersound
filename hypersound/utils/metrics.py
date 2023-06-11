from typing import Type

import librosa
import torch
from cdpam import CDPAM
from pesq import NoUtterancesError  # type: ignore
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.regression.mse import MeanSquaredError

METRICS = [
    "MAE",
    "MSE",
    "SNR",
    "SI-SNR",
    "PSNR",
    "LSD",
    "PESQ",
    "STOI",
    "CDPAM",
]


def reduce_metric(outputs: list[dict[str, Tensor]], key: str) -> float:
    try:
        values = [out[key] for out in outputs if not out[key].isnan()]
    except KeyError:
        return float("nan")
    if not values:
        return float("nan")
    return float(torch.stack(values).mean().detach())


def _get_metric(is_active: bool, metric: Type[Metric], preds: Tensor, target: Tensor) -> Tensor:
    if not is_active:
        return torch.tensor(float("nan"))

    _metric = metric(full_state_update=False).to(preds.device)  # type: ignore
    _metric.update(preds=preds, target=target)  # type: ignore
    return _metric.compute()  # type: ignore


def _resample(signal: Tensor, orig_sr: int, target_sr: int):
    device = signal.device

    signal = torch.tensor(
        librosa.resample(  # type: ignore
            signal.detach().cpu().numpy(),
            orig_sr=orig_sr,
            target_sr=target_sr,
        )
    )

    return signal.to(device)


def compute_metrics(
    preds: Tensor,
    target: Tensor,
    sample_rate: int,
    *,
    mae: bool = True,
    mse: bool = True,
    snr: bool = True,
    si_snr: bool = True,
    psnr: bool = True,
    lsd: bool = True,
    pesq: bool = False,
    stoi: bool = False,
    cdpam: bool = False,
) -> dict[str, Tensor]:

    results: dict[str, Tensor] = {}

    ScaleInvariantSignalNoiseRatio.full_state_update = False

    results["MSE"] = _get_metric(mse, MeanSquaredError, preds, target).detach()
    results["MAE"] = _get_metric(mae, MeanAbsoluteError, preds, target).detach()
    results["LSD"] = log_spectral_distance(lsd, preds, target, sample_rate=sample_rate).detach()
    results["SNR"] = _get_metric(snr, SignalNoiseRatio, preds, target).detach()
    results["PSNR"] = _get_metric(psnr, PeakSignalNoiseRatio, preds, target).detach()
    results["SI-SNR"] = _get_metric(si_snr, ScaleInvariantSignalNoiseRatio, preds, target).detach()

    results["PESQ"] = torch.tensor(float("nan"))
    if pesq:
        preds_16k = torch.stack([_resample(y, sample_rate, 16000) for y in preds])
        target_16k = torch.stack([_resample(y, sample_rate, 16000) for y in target])

        try:
            _pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
            _pesq.update(preds=preds_16k, target=target_16k)
            results["PESQ"] = _pesq.compute().detach()
        except NoUtterancesError:
            print("PESQ computation failed, skipping batch - no utterances found.")
            results["PESQ"] = torch.tensor(float("nan"))

    results["STOI"] = torch.tensor(float("nan"))
    if stoi:
        _stoi = ShortTimeObjectiveIntelligibility(fs=sample_rate)
        _stoi.update(preds=preds, target=target)
        results["STOI"] = _stoi.compute().detach()

    results["CDPAM"] = torch.tensor(float("nan"))
    if cdpam:
        _cdpam = CDPAM(dev=preds.device)
        results["CDPAM"] = compute_cdpam(_cdpam, preds=preds, target=target, sample_rate=sample_rate).detach()

    results = {metric: value.to("cpu") for metric, value in results.items()}

    return results


def log_spectral_distance(
    is_active: bool, preds: Tensor, target: Tensor, sample_rate: int, eps: float = 1e-12
) -> Tensor:
    """
    Log spectral distance between two spectrograms as per https://arxiv.org/pdf/2203.14941v1.pdf.
    `hop_length` and `n_fft` are computed as in the paper.
    """

    if not is_active:
        return torch.tensor(0.0)

    hop_length = int(sample_rate / 100)
    n_fft = int(2048 / (44100 / sample_rate))
    target_stft = _to_spectrogram(target, hop_length=hop_length, n_fft=n_fft)
    pred_stft = _to_spectrogram(preds, hop_length=hop_length, n_fft=n_fft)

    lsd = torch.log10(target_stft**2 / ((pred_stft + eps) ** 2) + eps) ** 2
    lsd = torch.mean(lsd, dim=-1) ** 0.5
    lsd = torch.mean(lsd, dim=-1)
    return lsd.mean()


def _to_spectrogram(audio: Tensor, hop_length: int, n_fft: int) -> Tensor:
    stft = torch.stft(
        audio,
        hop_length=hop_length,
        n_fft=n_fft,
        window=torch.hann_window(window_length=n_fft).to(audio.device),
        return_complex=True,
        pad_mode="constant",
    )
    stft = torch.abs(stft)
    stft = torch.transpose(stft, -1, -2)
    return stft


def compute_cdpam(
    model: CDPAM,
    preds: Tensor,
    target: Tensor,
    sample_rate: int,
) -> Tensor:
    """
    Computes CDPAM metric introduced in https://arxiv.org/abs/2102.05109.
    Requires CDPAM model, which operates on audio sampled with 22050 Hz.
    In case of different sampling rate, resamples recordings.
    """

    if sample_rate != 22050:
        preds = _resample(preds, sample_rate, 22050)
        target = _resample(target, sample_rate, 22050)

    preds = torch.round(preds.float() * 32768)
    target = torch.round(target.float() * 32768)

    with torch.no_grad():
        return model.forward(target, preds).detach().cpu().mean()  # type: ignore
