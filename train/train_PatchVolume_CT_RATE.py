import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE
from train.callbacks import VolumeLogger
from dataset.vqgan_4x_ct_rate import VQGANDataset_4x_CT_RATE
import argparse
from omegaconf import OmegaConf

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"  # Use NCCL for GPU training (faster than gloo)


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    pl.seed_everything(cfg.model.seed)
    downsample_ratio = cfg.model.downsample[0]
    
    if downsample_ratio == 4:
        train_dataset = VQGANDataset_4x_CT_RATE(
            root_dir=cfg.dataset.root_dir, augmentation=True, split='train', stage=cfg.model.stage)
        # Skip validation dataset since it's the same as training data
        val_dataset = VQGANDataset_4x_CT_RATE(
            root_dir=cfg.dataset.root_dir, augmentation=False, split='val')
    else:
        raise ValueError(f"Unsupported downsample ratio: {downsample_ratio}. Only 4x is supported for CT-RATE.")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size, shuffle=True,
                                  num_workers=cfg.model.num_workers)

    # Skip validation dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, lr, ngpu = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus

    print("Setting learning rate to {:.2e}, batch size to {}, ngpu to {}".format(lr, bs, ngpu))

    model = patchvolumeAE(cfg)

    callbacks = []
    # Change monitoring from validation loss to training loss since we skip validation
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=2000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(VolumeLogger(
        batch_frequency=1000, max_volumes=4, clamp=True))

    logger = TensorBoardLogger(cfg.model.default_root_dir, name="CT_RATE_model")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.model.gpus,
        default_root_dir=cfg.model.default_root_dir,
        strategy='ddp_find_unused_parameters_true',
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        # Disable validation completely
        check_val_every_n_epoch=2,
        num_sanity_val_steps=2,
        logger=logger
    )

    if cfg.model.resume_from_checkpoint and os.path.exists(cfg.model.resume_from_checkpoint):
        print('will start from the recent ckpt %s' % cfg.model.resume_from_checkpoint)
        # Skip validation dataloader - only use training data
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.model.resume_from_checkpoint)
    else:
        # Skip validation dataloader - only use training data
        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
