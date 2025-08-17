import os, argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from isic_datamodule import ISICInpaintDataModule
from model_inpaint import InpaintLightningModule

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--devices", type=str, default="auto")  # e.g., "1", "4"
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto") # "ddp" on multi-gpu
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--experiment", type=str, default="lightningdit-inpainting")
    p.add_argument("--mlflow_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)  # ckpt path
    return p.parse_args()

def main():
    args = parse_args()

    # Data
    dm = ISICInpaintDataModule(root=args.data_dir, size=args.size,
                               batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = InpaintLightningModule(lr=args.lr)

    # MLflow
    mlf = MLFlowLogger(experiment_name=args.experiment, tracking_uri=args.mlflow_uri, run_name=args.run_name)

    # Callbacks
    ckpt_cb = ModelCheckpoint(dirpath="checkpoints", save_last=True, save_top_k=2,
                              monitor="val/loss", mode="min", filename="epoch{epoch:02d}-valloss{val/loss:.4f}")
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        logger=mlf,
        default_root_dir=".",
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=1.0,
        log_every_n_steps=25
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
