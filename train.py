import copy
from pathlib import Path
from typing import Optional, Type, cast

import hydra
import pytorch_lightning as pl
import pytorch_yard
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_yard.configs import get_tags
from pytorch_yard.experiments.lightning import LightningExperiment
from pytorch_yard.utils.logging import info, info_bold

from hypersound.cfg import Settings
from hypersound.datasets.utils import init_datamodule
from hypersound.systems.main import HyperNetworkAE


class HyperSound(LightningExperiment):
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

        self.datamodule, _ = init_datamodule(self.root_cfg)
        self.datamodule.prepare_data()

        # --------------------------------------------------------------------
        # System setup
        # --------------------------------------------------------------------
        self.system = HyperNetworkAE(
            cfg=self.cfg,
            input_length=self.datamodule.train.shape[1][0],
            spec_transform=copy.deepcopy(self.datamodule.train.spec_transform),
        )

        info_bold("System architecture:")
        info(str(self.system))
        # info_bold(f"Size of target network: {cast(Any, self.system.target_network).num_params:,d}")

        info_bold(f"Input shape: {self.datamodule.shape}")

        # --------------------------------------------------------------------
        # Trainer setup
        # --------------------------------------------------------------------
        self.setup_callbacks()

        num_sanity_val_steps = -1 if self.cfg.validate_before_training else 0

        self.trainer: pl.Trainer = hydra.utils.instantiate(  # type: ignore
            self.cfg.pl,
            logger=self.wandb_logger,
            callbacks=self.callbacks,
            enable_checkpointing=self.cfg.save_checkpoints,
            num_sanity_val_steps=num_sanity_val_steps,
        )

        self.trainer.fit(  # type: ignore
            self.system,
            datamodule=self.datamodule,
            ckpt_path=self.cfg.resume_path,
        )


if __name__ == "__main__":
    HyperSound("hypersound", Settings)
