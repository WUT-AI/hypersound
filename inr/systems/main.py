import gc
from typing import Any, Iterable, Optional, Union, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler
import wandb
from pytorch_yard.lightning import LightningModuleWithWandb
from torch import Tensor
from torch.nn import L1Loss

# isort: split

from hypersound.cfg import ModelType, Settings, TargetNetworkMode
from hypersound.datasets.audio import IndicesAudioAndSpectrogram
from hypersound.models.meta.inr import INR
from hypersound.models.nerf import NERF
from hypersound.models.siren import SIREN
from hypersound.systems.loss import MultiSTFTLoss
from hypersound.utils.metrics import METRICS, compute_metrics
from hypersound.utils.wandb import fig_to_wandb


class INRSystem(LightningModuleWithWandb):
    """Functional representation of audio data learned example by example."""

    def __init__(self, cfg: Settings, spec_transform: nn.ModuleList, idx: int, extended_logging: bool):
        super().__init__()
        self.save_hyperparameters()  # type: ignore

        self.cfg = cfg
        """Main experiment config."""

        self.spec_transform = spec_transform
        """Raw signal to spectrogram transform provided as an `nn.ModuleList`."""

        self.inr: Union[NERF, SIREN]
        """INR network"""

        self.idx = idx
        """Example index, used for logging."""

        self.extended_logging = extended_logging
        """If True, will log reconstructions obtained with this system"""

        # ------------------------------------------------------------------------------------------

        if self.cfg.model.type is ModelType.SIREN:
            self.inr = SIREN(
                input_size=1,
                output_size=1,
                hidden_sizes=self.cfg.model.target_network_layer_sizes,
                bias=True,
                mode=TargetNetworkMode.INR,
                omega_0=self.cfg.model.target_network_omega_0,
                omega_i=self.cfg.model.target_network_omega_i,
                learnable_omega=self.cfg.model.target_network_learnable_omega,
                gradient_fix=self.cfg.model.target_network_siren_gradient_fix,
            )

        elif self.cfg.model.type is ModelType.NERF:
            self.inr = NERF(
                input_size=1,
                output_size=1,
                hidden_sizes=self.cfg.model.target_network_layer_sizes,
                bias=True,
                mode=TargetNetworkMode.INR,
                encoding_length=self.cfg.model.target_network_encoding_length,
                learnable_encoding=self.cfg.model.target_network_learnable_encoding,
            )
        else:
            raise ValueError(f"Unknown type of INR: {self.cfg.model.type}")

        self.reconstruction_loss = L1Loss()

        self.perceptual_loss = MultiSTFTLoss(
            sample_rate=self.cfg.data.sample_rate,
            fft_sizes=self.cfg.model.fft_sizes,
            hop_sizes=self.cfg.model.hop_sizes,
            win_lengths=self.cfg.model.win_lengths,
            n_bins=self.cfg.data.n_mels,
            freq_weights_warmup_epochs=cfg.model.perceptual_loss_freq_weights_warmup_epochs,
            freq_weights_p=self.cfg.model.perceptual_loss_freq_weights_p
        )

        self.data: Optional[IndicesAudioAndSpectrogram] = None
        self.metrics: dict[str, Tensor] = {}

        self._activations: list[tuple[Tensor, Tensor]]

    def compression_ratio(self) -> float:
        if self.data is not None:
            indices, _, _ = self.data
        else:
            raise ValueError("You must run at least one training step to get the compression rate.")
        num_params = cast(INR, self.inr).num_params()
        num_samples = indices.numel()
        return num_samples / num_params

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self._log_inr(init_log=True)

    def forward(  # type: ignore
        self,
        indices: Tensor,
        audio: Tensor,
        spectrogram: Tensor,
        log_reconstructions: bool = False,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Reconstruct a spectrogram with.

        Optionally, log examples to wandb.

        Parameters
        ----------
        indices : Tensor
            Time indices.
        audio : Tensor
            Audio samples at given indices.
        spectrogram: Tensor
            Spectrogram data (full length).
        log_reconstructions: Whether to log reconstructions.
        Returns
        -------
        dict[str, Tensor]
            Loss values.

        Tensor
            Audio reconstructions.

        """
        # Audio shape should be (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1
        assert audio.shape[1] > 1

        # Indices shape should be (num_samples, 1)
        assert len(indices.shape) == 3
        assert indices.shape[0] == 1
        indices = indices.squeeze(dim=0)
        assert indices.shape[0] == audio.shape[1]
        assert indices.shape[1] == 1

        # Spectrogram shape should be (1, num_mels, num_bins)
        assert len(spectrogram.shape) == 4
        assert spectrogram.shape[0] == 1
        spectrogram = spectrogram.squeeze(dim=0)
        assert spectrogram.shape[0] == 1
        assert spectrogram.shape[1] > 1
        assert spectrogram.shape[2] > 1

        if log_reconstructions:
            audio_reconstruction, self._activations = self.inr(indices, return_activations=True)
        else:
            audio_reconstruction = self.inr(indices)

        audio_reconstruction = audio_reconstruction.squeeze(dim=1).unsqueeze(dim=0)
        assert audio_reconstruction.shape == audio.shape

        spectrogram_reconstruction = self.to_spectrogram(audio_reconstruction)
        assert spectrogram_reconstruction.shape == spectrogram.shape

        reconstruction_loss = self.reconstruction_loss(audio, audio_reconstruction)
        perceptual_loss = self.perceptual_loss(audio, audio_reconstruction)

        if log_reconstructions:
            self.log_reconstructions(
                audio=audio,
                audio_reconstructions=audio_reconstruction,
                spectrograms=spectrogram,
                spectrogram_reconstructions=spectrogram_reconstruction,
            )

        total_loss = (
            self.cfg.model.reconstruction_loss_lambda * reconstruction_loss
            + self.cfg.model.perceptual_loss_lambda * perceptual_loss
        )

        return (
            dict(
                loss=total_loss,
                reconstruction=reconstruction_loss.detach(),
                perceptual=perceptual_loss.detach(),
            ),
            audio_reconstruction,
        )

    def _on_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:  # type: ignore
        metrics = {
            f"loss/total/r{self.idx}": outputs[-1]["loss"],
            f"loss/reconstruction/r{self.idx}": outputs[-1]["reconstruction"],
            f"loss/perceptual/r{self.idx}": outputs[-1]["perceptual"],
            "epoch": self.epoch,
        }

        for metric in METRICS:
            if metric in outputs[-1]:
                metrics[f"metric/{metric}/r{self.idx}"] = outputs[-1][metric]

        self.log_wandb(
            metrics,
        )

        if self.extended_logging:
            with self.no_train():
                self.forward(*cast(IndicesAudioAndSpectrogram, self.data), log_reconstructions=True)

    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: IndicesAudioAndSpectrogram, batch_idx: int) -> dict[str, Tensor]:  # type: ignore
        if self.data is None:
            self.data = batch

        is_last = batch_idx == cast(Any, self.trainer).num_training_batches - 1

        loss, reconstructions = self.forward(*batch)

        metrics = compute_metrics(
            reconstructions,
            batch[1],
            sample_rate=self.cfg.data.sample_rate,
            pesq=is_last,
            stoi=is_last,
            cdpam=is_last,
        )
        loss.update(metrics)

        self.metrics = loss

        return loss

    def training_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:  # type: ignore
        self._on_epoch_end(outputs=outputs)

    # Optimizers
    # ----------------------------------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore
        # Main model optimization
        optimizer_main = torch.optim.Adam(
            self.inr.parameters(),
            lr=self.cfg.learning_rate,
        )

        scheduler_ = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_main, lr_lambda=lambda _: 1.0)

        optimizers = [optimizer_main]
        schedulers = [scheduler_]

        return optimizers, schedulers  # type: ignore

    # Helpers
    # ----------------------------------------------------------------------------------------------
    def to_spectrogram(self, y: Tensor) -> Tensor:
        """Convert raw audio signal to spectrogram using the pre-defined transform.

        Parameters
        ----------
        y : Tensor
            Audio signal.

        Returns
        -------
        Tensor
            Spectrogram.

        """
        for transform in self.spec_transform:
            y = transform(y)
        return y

    # Visualizations
    # ----------------------------------------------------------------------------------------------
    def log_reconstructions(
        self,
        audio: Tensor,
        audio_reconstructions: Tensor,
        spectrograms: Tensor,
        spectrogram_reconstructions: Tensor,
    ) -> None:
        """Log reconstruction samples to wandb.

        Parameters
        ----------
        audio : Tensor
            [description]
        audio_reconstructions : Tensor
            [description]
        spectrograms : Tensor
            [description]
        spectrogram_reconstructions : Tensor
            [description]

        """
        self._log_as_imgs(
            audio,
            audio_reconstructions,
            spectrograms,
            spectrogram_reconstructions,
        )
        self._log_audio(audio, audio_reconstructions)
        self._log_inr()

    def _log_as_imgs(
        self,
        audio: Tensor,
        audio_reconstructions: Tensor,
        spectrograms: Tensor,
        spectrogram_reconstructions: Tensor,
    ):
        assert len(audio) == len(audio_reconstructions) == len(spectrograms) == len(spectrogram_reconstructions)

        with plt.style.context("bmh"):  # type: ignore
            fig, axes = plt.subplots(  # type: ignore
                nrows=4, ncols=len(audio), figsize=(3 * len(audio), 6), squeeze=False
            )
            fig.subplots_adjust(hspace=0.01, wspace=0.01)
            axes = cast(Any, axes)

            for i, ax in enumerate(axes[0]):
                ax.imshow(spectrograms[i].squeeze().detach().cpu().numpy(), origin="lower", aspect="auto")
                ax.set_axis_off()
            for i, ax in enumerate(axes[1]):
                ax.imshow(
                    spectrogram_reconstructions[i].squeeze().detach().cpu().numpy(), origin="lower", aspect="auto"
                )
                ax.set_axis_off()
            for i, ax in enumerate(axes[2]):
                ax.plot(audio[i].detach().cpu().numpy())
                ax.set_ylim(-1, 1)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
            for i, ax in enumerate(axes[3]):
                ax.plot(audio_reconstructions[i].detach().cpu().numpy())
                ax.set_ylim(-1, 1)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            self.log_wandb(
                {
                    f"reconstruction/r{self.idx}": fig_to_wandb(fig),
                    "epoch": self.epoch,
                }
            )
            plt.close("all")  # type: ignore
            gc.collect()

    def _log_audio(
        self,
        audio: Tensor,
        audio_reconstructions: Tensor,
    ):
        for i, (_audio, _reconstruction) in enumerate(
            zip(
                cast(Iterable[Tensor], audio),
                cast(Iterable[Tensor], audio_reconstructions),
            )
        ):
            self.log_wandb(
                {
                    f"audio/in/{i}/r{self.idx}": wandb.Audio(
                        _audio.detach().cpu().numpy(), sample_rate=self.cfg.data.sample_rate
                    ),
                    "epoch": self.epoch,
                },
            )
            self.log_wandb(
                {
                    f"audio/out/{i}/r{self.idx}": wandb.Audio(
                        _reconstruction.detach().cpu().numpy(), sample_rate=self.cfg.data.sample_rate
                    ),
                    "epoch": self.epoch,
                },
            )

    def _log_inr(self, init_log: bool = False):
        for layer in self.inr.params:
            if layer[0] not in ["w", "b"]:
                continue
            self.log_wandb(
                {
                    f"layer/{layer}/r{self.idx}": wandb.Histogram(self.inr.params[layer].detach().cpu().numpy()),
                    "epoch": self.epoch if not init_log else 0,
                },
            )

        if not init_log:
            for i, (activations, preactivations) in enumerate(self._activations):
                self.log_wandb(
                    {
                        f"activations/{i}/r{self.idx}": wandb.Histogram(activations.detach().cpu().numpy()),
                        f"preactivations/{i}/r{self.idx}": wandb.Histogram(preactivations.detach().cpu().numpy()),
                        "epoch": self.epoch,
                    },
                )
