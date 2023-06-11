import copy
from pathlib import Path
from typing import Optional, Type, cast

import hydra
import pytorch_lightning as pl
import pytorch_yard
import torch
import torch.utils.data
from dotenv import load_dotenv  # type: ignore
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_yard.configs import get_tags
from pytorch_yard.experiments.lightning import LightningExperiment
from torch import Tensor
from torch.utils.data import RandomSampler, TensorDataset

from hypersound.cfg import Settings
from hypersound.datasets.utils import init_datamodule
from hypersound.utils.metrics import reduce_metric
from inr.systems.main import INRSystem


class SingleINRExperiment(LightningExperiment):
    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        super().__init__(config_path, settings_cls, settings_group=settings_group)

        self.cfg: Settings
        """ Experiment config. """

    def entry(self, root_cfg: pytorch_yard.RootConfig):
        super().entry(root_cfg)

    # Do not use pytorch-yard template specializations as we use a monolithic `main` here.
    def setup_system(self):
        pass

    def setup_datamodule(self):
        pass

    # ------------------------------------------------------------------------
    # Experiment specific code
    # ------------------------------------------------------------------------
    def main(self):
        # --------------------------------------------------------------------
        # W&B init
        # --------------------------------------------------------------------
        tags: list[str] = get_tags(cast(DictConfig, self.root_cfg))
        self.run.tags = tags
        self.run.notes = str(self.root_cfg.notes)
        self.wandb_logger.log_hyperparams(OmegaConf.to_container(self.root_cfg.cfg, resolve=True))  # type: ignore

        # --------------------------------------------------------------------
        # Data module setup
        # --------------------------------------------------------------------
        Path(self.root_cfg.data_dir).mkdir(parents=True, exist_ok=True)

        self.root_cfg.cfg = cast(Settings, self.root_cfg.cfg)
        self.root_cfg.cfg.batch_size = 1
        self.root_cfg.cfg.save_checkpoints = False

        self.datamodule, _ = init_datamodule(self.root_cfg)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        # --------------------------------------------------------------------
        # Trainer setup
        # --------------------------------------------------------------------
        self.setup_callbacks()

        steps_to_log = range(self.cfg.log.examples)
        combined_metrics: list[dict[str, Tensor]] = []

        for i, (indices, audio, spectrograms) in enumerate(self.datamodule.val_dataloader()):
            callbacks = copy.deepcopy(self.callbacks)

            self.trainer: pl.Trainer = hydra.utils.instantiate(  # type: ignore
                self.cfg.pl,
                logger=self.wandb_logger,
                callbacks=callbacks,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
            )

            indices = torch.cat([indices])
            audio = torch.cat([audio])
            spectrograms = torch.cat([spectrograms])
            dataset = TensorDataset(indices, audio, spectrograms)

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                sampler=RandomSampler(dataset, replacement=True, num_samples=self.cfg.data.samples_per_epoch),
                num_workers=self.cfg.data.num_workers,
            )

            log_reconstruction = i in steps_to_log

            self.system = INRSystem(
                cfg=self.cfg,
                spec_transform=copy.deepcopy(self.datamodule.train.spec_transform),
                idx=i,
                extended_logging=log_reconstruction,
            )
            self.trainer.fit(  # type: ignore
                self.system,
                train_dataloaders=dataloader,
                ckpt_path=None,
            )
            combined_metrics.append(self.system.metrics)

            assert isinstance(self.system, INRSystem)
            self.wandb_logger.experiment.summary["combined_metrics/compression_ratio"] = self.system.compression_ratio()  # type: ignore # noqa
            self.wandb_logger.experiment.summary["combined_metrics/inr_idx"] = i + 1  # type: ignore
            for key in combined_metrics[0]:
                self.wandb_logger.experiment.summary[f"combined_metrics/{key}"] = reduce_metric(combined_metrics, key)  # type: ignore # noqa


if __name__ == "__main__":
    load_dotenv(".env.inr", verbose=True, override=True)
    SingleINRExperiment("hypersound", Settings)
