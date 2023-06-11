# fmt: off
import os
from os import environ, path

import GPUtil as gpu
import numpy as np
import pytorch_lightning as pl
import torch
from effortless_config import Config, setting
from rave.core import EMAModelCheckPoint, random_phase_mangle, search_for_run
from rave.model import RAVE
from torch.utils.data import DataLoader, random_split
from udls import SimpleDataset, simple_audio_preprocess
from udls.transforms import Compose, Dequantize, RandomApply, RandomCrop

if __name__ == "__main__":

    class args(Config):
        groups = ["small", "large"]

        DATA_SIZE = 16
        CAPACITY = setting(default=64, small=32, large=64)
        LATENT_SIZE = 128
        BIAS = True
        NO_LATENCY = False
        RATIOS = setting(
            default=[4, 4, 4, 2],
            small=[4, 4, 4, 2],
            large=[4, 4, 2, 2, 2],
        )

        MIN_KL = 1e-1
        MAX_KL = 1e-1
        CROPPED_LATENT_SIZE = 0
        FEATURE_MATCH = True

        LOUD_STRIDE = 1

        USE_NOISE = True
        NOISE_RATIOS = [4, 4, 4]
        NOISE_BANDS = 5

        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4

        WARMUP = setting(default=1000000, small=1000000, large=3000000)
        MODE = "hinge"
        CKPT = None

        PREPROCESSED = None
        WAV = None
        WAV_VAL = None
        SR = 48000
        N_SIGNAL = 65536
        MAX_STEPS = setting(default=3000000, small=3000000, large=6000000)
        VAL_EVERY = 10000

        BATCH = 8

        NAME = None

    args.parse_args()

    assert args.NAME is not None
    model = RAVE(
        data_size=args.DATA_SIZE,
        capacity=args.CAPACITY,
        latent_size=args.LATENT_SIZE,
        ratios=args.RATIOS,
        bias=args.BIAS,
        loud_stride=args.LOUD_STRIDE,
        use_noise=args.USE_NOISE,
        noise_ratios=args.NOISE_RATIOS,
        noise_bands=args.NOISE_BANDS,
        d_capacity=args.D_CAPACITY,
        d_multiplier=args.D_MULTIPLIER,
        d_n_layers=args.D_N_LAYERS,
        warmup=args.WARMUP,
        mode=args.MODE,
        no_latency=args.NO_LATENCY,
        sr=args.SR,
        min_kl=args.MIN_KL,
        max_kl=args.MAX_KL,
        cropped_latent_size=args.CROPPED_LATENT_SIZE,
        feature_match=args.FEATURE_MATCH,
    )

    x = torch.zeros(args.BATCH, 2**14)
    model.validation_step(x, 0)

    preprocess = lambda name: simple_audio_preprocess(args.SR, 2 * args.N_SIGNAL,)(
        name
    ).astype(np.float16)

    def get_dataset(out_folder, folder):
        return SimpleDataset(
            out_folder,
            folder,
            preprocess_function=preprocess,
            split_set="full",
            transforms=Compose([
                lambda x: x.astype(np.float32),
                RandomCrop(args.N_SIGNAL),
                RandomApply(
                    lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
                    p=.8,
                ),
                Dequantize(16),
                lambda x: x.astype(np.float32),
            ]),
        )

    train = get_dataset(path.join(args.PREPROCESSED, "train"), args.WAV)
    val = get_dataset(path.join(args.PREPROCESSED + "val"), args.WAV_VAL)

    # val = max((2 * len(dataset)) // 100, 1)
    # train = len(dataset) - val
    # train, val = random_split(
    #     dataset,
    #     [train, val],
    #     generator=torch.Generator().manual_seed(42),
    # )

    num_workers = 0 if os.name == "nt" else 8
    train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=num_workers)
    val = DataLoader(val, args.BATCH, False, num_workers=num_workers)

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="validation",
        filename="best",
    )
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    CUDA = gpu.getAvailable(maxMemory=0.05)
    VISIBLE_DEVICES = environ.get("CUDA_VISIBLE_DEVICES", "")

    if VISIBLE_DEVICES:
        use_gpu = int(int(VISIBLE_DEVICES) >= 0)
    elif len(CUDA):
        environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
        use_gpu = 1
    elif torch.cuda.is_available():
        print("Cuda is available but no fully free GPU found.")
        print("Training may be slower due to concurrent processes.")
        use_gpu = 1
    else:
        print("No GPU found.")
        use_gpu = 0

    val_check = {}
    if len(train) >= args.VAL_EVERY:
        val_check["val_check_interval"] = args.VAL_EVERY
    else:
        nepoch = args.VAL_EVERY // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(path.join("runs", args.NAME),
                                            name="rave"),
        gpus=use_gpu,
        callbacks=[validation_checkpoint, last_checkpoint],
        max_epochs=100000,
        max_steps=args.MAX_STEPS,
        **val_check,
    )

    run = search_for_run(args.CKPT, mode="last")
    if run is None: run = search_for_run(args.CKPT, mode="best")
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    trainer.fit(model, train, val, ckpt_path=run)
