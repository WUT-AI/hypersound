import gc
from contextlib import suppress
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Any, Iterable, Union, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import wandb
import wandb.plot
from omegaconf import OmegaConf
from pytorch_yard.lightning import LightningModuleWithWandb
from torch import Tensor
from torch.nn import L1Loss
from torch.utils.data import DataLoader, Subset

# isort: split

from hypersound.cfg import LRScheduler, ModelType, Settings, TargetNetworkMode
from hypersound.datamodules.audio import AudioDataModule
from hypersound.datasets.audio import IndicesAudioAndSpectrogram
from hypersound.models.encoder import Encoder
from hypersound.models.meta.hyper import MLPHyperNetwork
from hypersound.models.meta.inr import INR
from hypersound.models.nerf import NERF
from hypersound.models.siren import SIREN
from hypersound.systems.loss import MultiSTFTLoss
from hypersound.utils.metrics import METRICS, compute_metrics, reduce_metric
from hypersound.utils.wandb import fig_to_wandb


class HyperNetworkAE(LightningModuleWithWandb):
    """Functional representation of audio recordings with a hypernetwork-autoencoder."""

    def __init__(
        self,
        cfg: Settings,
        input_length: int,
        spec_transform: nn.ModuleList,
    ):
        super().__init__()
        self.save_hyperparameters()  # type: ignore

        self.cfg = cfg
        """Main experiment config."""

        self.spec_transform = spec_transform
        """Raw signal to spectrogram transform provided as an `nn.ModuleList`."""

        self.encoder: Encoder
        """Encoder network."""

        self.hypernetwork: MLPHyperNetwork
        """Hypernetwork."""

        self.target_network: INR
        """Target network"""

        self.examples: dict[str, DataLoader[IndicesAudioAndSpectrogram]] = {}
        """Data loaders for plotting examples."""

        # ------------------------------------------------------------------------------------------
        self.encoder = Encoder(
            C=self.cfg.model.encoder_channels,
            D=self.cfg.model.embedding_size,
            **cast(dict[str, Any], OmegaConf.to_container(self.cfg.model, resolve=True)),
        )

        self.target_network, shared_params = self._init_target_network()

        self.hypernetwork = MLPHyperNetwork(
            target_network=self.target_network,
            shared_params=shared_params,
            input_size=self.encoder.output_width(input_length) * self.cfg.model.embedding_size,
            layer_sizes=self.cfg.model.hypernetwork_layer_sizes,
        )

        if self.cfg.model.target_network_mode is TargetNetworkMode.TARGET_NETWORK:
            self.target_network.freeze_params(shared_params=shared_params)

        # ------------------------------------------------------------------------------------------
        self.reconstruction_loss = L1Loss()

        self.perceptual_loss = MultiSTFTLoss(
            sample_rate=self.cfg.data.sample_rate,
            fft_sizes=self.cfg.model.fft_sizes,
            hop_sizes=self.cfg.model.hop_sizes,
            win_lengths=self.cfg.model.win_lengths,
            n_bins=self.cfg.data.n_mels,
            freq_weights_warmup_epochs=cfg.model.perceptual_loss_freq_weights_warmup_epochs,
            freq_weights_p=self.cfg.model.perceptual_loss_freq_weights_p,
        )

        # ------------------------------------------------------------------------------------------
        self._weights: dict[str, Tensor]
        self._activations: list[tuple[Tensor, Tensor]]

        self._log_init: bool = True

    def _gc(self) -> None:
        attrs = "_weights", "_activations"

        for attr in attrs:
            with suppress(AttributeError):
                delattr(self, attr)

    def _init_target_network(self) -> tuple[INR, list[str]]:
        target_network: Union[NERF, SIREN]
        shared_params: list[str] = []

        if self.cfg.model.type is ModelType.NERF:
            target_network = NERF(
                input_size=1,
                output_size=1,
                hidden_sizes=self.cfg.model.target_network_layer_sizes,
                bias=True,
                mode=self.cfg.model.target_network_mode,
                encoding_length=self.cfg.model.target_network_encoding_length,
                learnable_encoding=self.cfg.model.target_network_learnable_encoding,
            )
            if self.cfg.model.target_network_share_encoding:
                shared_params.append("freq")

        elif self.cfg.model.type is ModelType.SIREN:
            target_network = SIREN(
                input_size=1,
                output_size=1,
                hidden_sizes=self.cfg.model.target_network_layer_sizes,
                bias=True,
                mode=self.cfg.model.target_network_mode,
                omega_0=self.cfg.model.target_network_omega_0,
                omega_i=self.cfg.model.target_network_omega_i,
                learnable_omega=self.cfg.model.target_network_learnable_omega,
                gradient_fix=self.cfg.model.target_network_siren_gradient_fix,
            )
            if self.cfg.model.target_network_share_omega:
                shared_params.extend(
                    [param_name for param_name in target_network.params.keys() if "o" in param_name]
                )  # FIXME: if param_name[0] == "o"?
        else:
            raise RuntimeError(f"Unknown model type: {self.cfg.model.type}")

        if (
            self.cfg.model.target_network_mode is TargetNetworkMode.TARGET_NETWORK
            and self.cfg.model.target_network_shared_layers is not None
        ):
            for i in self.cfg.model.target_network_shared_layers:
                shared_params.append(f"w{i}")
                shared_params.append(f"b{i}")

        return target_network, shared_params

    # Setup
    # ----------------------------------------------------------------------------------------------
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        last_layer_idx = len(self.hypernetwork.net) - 1
        bias_key = f"hypernetwork.net.{last_layer_idx}.bias"

        if bias_key not in checkpoint["state_dict"]:
            biases = cast(list[nn.Linear], self.hypernetwork.net)[-1].bias
            checkpoint["state_dict"][bias_key] = torch.zeros_like(biases)

            last_opt_key = list(checkpoint["optimizer_states"][0]["state"].keys())[-1]

            checkpoint["optimizer_states"][0]["state"][last_opt_key + 1] = {
                "step": checkpoint["optimizer_states"][0]["state"][last_opt_key]["step"],
                "exp_avg": torch.zeros_like(biases, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(biases, memory_format=torch.preserve_format),
            }
            checkpoint["optimizer_states"][0]["param_groups"][0]["params"].append(last_opt_key + 1)

            if self.cfg.scheduler is LRScheduler.CONSTANT:
                checkpoint["lr_schedulers"][0] = {
                    "base_lrs": checkpoint["lr_schedulers"][0]["base_lrs"],
                    "last_epoch": checkpoint["lr_schedulers"][0]["last_epoch"],
                    "_step_count": checkpoint["lr_schedulers"][0]["_step_count"],
                    "verbose": checkpoint["lr_schedulers"][0]["verbose"],
                    "_get_lr_called_within_step": checkpoint["lr_schedulers"][0]["_get_lr_called_within_step"],
                    "_last_lr": checkpoint["lr_schedulers"][0]["base_lrs"],
                    "lr_lambdas": [None],
                }

        return super().on_load_checkpoint(checkpoint)

    def on_fit_start(self) -> None:
        """Prepare data loaders for logging examples to wandb."""
        super().on_fit_start()

        # Prepare data loaders for logging examples
        dm = deepcopy(self.trainer.datamodule)  # type: ignore
        assert isinstance(dm, AudioDataModule)
        assert dm.train and dm.validation

        for phase in ["train", "validation"]:
            dataset = getattr(dm, phase)
            rng = torch.Generator().manual_seed(20211026)
            idxs = torch.randperm(len(dataset), generator=rng)[: self.cfg.log.examples]
            examples = Subset(dataset, idxs.tolist())  # type: ignore

            self.examples[phase] = DataLoader(
                examples, batch_size=self.cfg.log.examples, shuffle=False, pin_memory=True
            )

    # Forward
    # ----------------------------------------------------------------------------------------------
    def reconstruct(self, indices: Tensor, audio: Tensor, save_activations: bool = False) -> Tensor:
        z = self.encoder(audio)

        assert z.shape[1] == self.encoder.output_width(audio.shape[1]) * self.cfg.model.embedding_size

        # HyperNet reconstruction
        self._weights = self.hypernetwork(z)

        if save_activations:
            audio_reconstructions, self._activations = self.target_network(
                x=indices,
                weights=self._weights,
                return_activations=True,
            )
        else:
            audio_reconstructions = self.target_network(x=indices, weights=self._weights)

        audio_reconstructions = audio_reconstructions.squeeze(dim=-1)

        return audio_reconstructions

    def main_loss(self, audio: Tensor, audio_reconstructions: Tensor) -> dict[str, Tensor]:  # type: ignore  # noqa
        reconstruction_loss: Tensor = self.reconstruction_loss(audio, audio_reconstructions)
        perceptual_loss: Tensor = self.perceptual_loss(audio, audio_reconstructions)

        if self.cfg.model.perceptual_loss_decay_epochs is not None:
            if self.epoch < self.cfg.model.perceptual_loss_decay_epochs:
                decay_factor = (
                    self.cfg.model.perceptual_loss_decay_epochs - self.current_epoch
                ) / self.cfg.model.perceptual_loss_decay_epochs
                perceptual_loss_lambda = (
                    decay_factor * self.cfg.model.perceptual_loss_lambda
                    + (1 - decay_factor) * self.cfg.model.perceptual_loss_final_lambda
                )
            else:
                perceptual_loss_lambda = self.cfg.model.perceptual_loss_final_lambda
        else:
            perceptual_loss_lambda = self.cfg.model.perceptual_loss_lambda

        total_loss = (
            self.cfg.model.reconstruction_loss_lambda * reconstruction_loss  # nofmt
            + perceptual_loss_lambda * perceptual_loss
        )

        return dict(
            loss=total_loss,
            reconstruction=reconstruction_loss.detach().cpu(),
            perceptual=perceptual_loss.detach().cpu(),
        )

    def _on_epoch_end(
        self,
        phase: str,
        outputs: Union[list[dict[str, Tensor]], list[list[dict[str, Tensor]]]],
        log_reconstructions: bool = True,
    ) -> None:  # type: ignore
        if isinstance(outputs[0], list):
            outputs = cast(list[list[dict[str, Tensor]]], outputs)

            merged: list[dict[str, Tensor]] = []

            for i in range(len(outputs[0])):
                merged.append(dict())
                for optimizer_out in outputs:
                    merged[-1].update(optimizer_out[i])

            outputs = merged

        outputs = cast(list[dict[str, Tensor]], outputs)

        metrics = {
            f"loss/{phase}/total": reduce_metric(outputs, "loss"),
            f"loss/{phase}/reconstruction": reduce_metric(outputs, "reconstruction"),
            f"loss/{phase}/perceptual": reduce_metric(outputs, "perceptual"),
            "epoch": self.epoch,
        }
        for metric in METRICS:
            if metric in outputs[0]:
                metrics[f"metric/{phase}/{metric}"] = reduce_metric(outputs, metric)

        self.log_wandb(
            metrics,
            step=self.epoch,
        )

        if log_reconstructions:
            assert phase in self.examples

            with self.no_train():
                indices, audio, spectrograms = cast(
                    IndicesAudioAndSpectrogram, [b.to(self.device) for b in next(iter(self.examples[phase]))]
                )

                save_activations = phase == "train"
                audio_reconstructions = self.reconstruct(indices, audio, save_activations=save_activations)

                spectrogram_reconstructions = self.to_spectrogram(audio_reconstructions)
                assert spectrogram_reconstructions.shape == spectrograms.shape

                self.log_reconstructions(
                    log_as=phase,
                    audio=audio,
                    audio_reconstructions=audio_reconstructions,
                    spectrograms=spectrograms,
                    spectrogram_reconstructions=spectrogram_reconstructions,
                )
        self.perceptual_loss.update(self.epoch + 1)

        self._gc()

    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: IndicesAudioAndSpectrogram, batch_idx: int) -> dict[str, Tensor]:  # type: ignore
        indices, audio, _ = batch

        audio_reconstructions = self.reconstruct(indices, audio, save_activations=self._log_init)
        loss = self.main_loss(audio, audio_reconstructions)

        if self._log_init:
            self._log_target_network(log_init=True)
            self._log_init = False

        self._gc()

        return loss

    def training_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:  # type: ignore
        if self.epoch < self.cfg.log.warmup_epochs:
            log_reconstructions = self.epoch % self.cfg.log.warmup_every_n_epoch == 0
        else:
            log_reconstructions = self.epoch % self.cfg.log.normal_every_n_epoch == 0
        self._on_epoch_end(phase="train", outputs=outputs, log_reconstructions=log_reconstructions)

    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: IndicesAudioAndSpectrogram, batch_idx: int) -> dict[str, Tensor]:  # type: ignore
        indices, audio, _ = batch

        audio_reconstructions = self.reconstruct(indices, audio)

        loss = self.main_loss(audio, audio_reconstructions)
        metrics = compute_metrics(
            audio_reconstructions, batch[1], sample_rate=self.cfg.data.sample_rate, pesq=True, stoi=True, cdpam=True
        )
        loss.update(metrics)

        self._gc()

        return loss

    def validation_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:  # type: ignore
        self._on_epoch_end(phase="validation", outputs=outputs, log_reconstructions=True)

    # Optimizers
    # ----------------------------------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore
        # Main model optimization
        optimizer_main = torch.optim.AdamW(
            chain(self.encoder.parameters(), self.target_network.parameters(), self.hypernetwork.parameters()),
            lr=self.cfg.learning_rate,
        )

        steps_per_epoch = int(self.cfg.data.samples_per_epoch / self.cfg.batch_size)

        if self.cfg.scheduler is LRScheduler.ONE_CYCLE:
            scheduler_: partial[Any] = partial(
                torch.optim.lr_scheduler.OneCycleLR,  # type: ignore
                max_lr=self.cfg.learning_rate * self.cfg.learning_rate_div_factor,
                div_factor=self.cfg.learning_rate_div_factor,
                epochs=self.cfg.pl.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.05,
            )
        elif self.cfg.scheduler is LRScheduler.CONSTANT:
            scheduler_: partial[Any] = partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda _: 1.0)  # type: ignore  # noqa
        else:
            raise NotImplementedError(f"Unknown scheduler {self.cfg.scheduler}")

        optimizers = [optimizer_main]
        schedulers = [scheduler_(optimizer_main)]

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
        y = y.unsqueeze(-3)
        return y

    # Visualizations
    # ----------------------------------------------------------------------------------------------
    def log_reconstructions(
        self,
        log_as: str,
        audio: Tensor,
        audio_reconstructions: Tensor,
        spectrograms: Tensor,
        spectrogram_reconstructions: Tensor,
    ) -> None:
        """Log reconstruction samples to wandb.

        Parameters
        ----------
        log_as : str
            [description]
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
            log_as,
            audio,
            audio_reconstructions,
            spectrograms,
            spectrogram_reconstructions,
        )
        self._log_audio(log_as, audio, audio_reconstructions)
        if log_as == "train":
            self._log_target_network()

    def _log_as_imgs(
        self,
        log_as: str,
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

            self.log_wandb({f"reconstruction/{log_as}": fig_to_wandb(fig)}, step=self.epoch)
            plt.close("all")  # type: ignore
            gc.collect()

    def _log_audio(
        self,
        log_as: str,
        audio: Tensor,
        audio_reconstructions: Tensor,
    ):
        for idx, (audio, reconstruction) in enumerate(
            zip(
                cast(Iterable[Tensor], audio),
                cast(Iterable[Tensor], audio_reconstructions),
            )
        ):
            self.log_wandb(
                {
                    f"audio/{log_as}/in/{idx}": wandb.Audio(
                        audio.detach().cpu().numpy(), sample_rate=self.cfg.data.sample_rate
                    )
                },
                step=self.epoch,
            )
            self.log_wandb(
                {
                    f"audio/{log_as}/out/{idx}": wandb.Audio(
                        reconstruction.detach().cpu().numpy(), sample_rate=self.cfg.data.sample_rate
                    )
                },
                step=self.epoch,
            )

    def _log_target_network(self, log_init: bool = False):
        if not self.cfg.log.debug:
            return

        data: dict[str, Any] = {}

        for i in range(self.target_network.n_layers):
            if f"w{i}" in self._weights:
                data[f"layer/w{i}"] = wandb.Histogram(self._weights[f"w{i}"].detach().cpu().numpy())
                if self.cfg.log.debug:
                    self.log_wandb(
                        {
                            f"hist/w{i}": wandb.plot.histogram(
                                wandb.Table(
                                    data=[[w] for w in self._weights[f"w{i}"].detach().cpu().numpy().flatten()],
                                    columns=[f"w{i}"],
                                ),
                                f"w{i}",
                            )
                        },
                        step=self.epoch if not log_init else 0,
                    )
            if f"b{i}" in self._weights:
                data[f"layer/b{i}"] = wandb.Histogram(self._weights[f"b{i}"].detach().cpu().numpy())
                if self.cfg.log.debug:
                    self.log_wandb(
                        {
                            f"hist/b{i}": wandb.plot.histogram(
                                wandb.Table(
                                    data=[[b] for b in self._weights[f"b{i}"].detach().cpu().numpy().flatten()],
                                    columns=[f"b{i}"],
                                ),
                                f"b{i}",
                            )
                        },
                        step=self.epoch if not log_init else 0,
                    )

        for i, (activations, preactivations) in enumerate(self._activations):
            data[f"activations/{i}"] = wandb.Histogram(activations.detach().cpu().numpy())
            data[f"preactivations/{i}"] = wandb.Histogram(preactivations.detach().cpu().numpy())

        if log_init:
            data["epoch"] = 0

        self.log_wandb(data, step=self.epoch if not log_init else 0)
