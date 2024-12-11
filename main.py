# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
from argparse import ArgumentParser
from typing import Type
import os
import torch
import torch.optim as optim
from dacite import from_dict
from data.formal_language.formal_language_dataset import (
    FormLangDatasetGenerator,
)
from data.utils import DataGen
from lr_scheduler import LinearWarmupCosineAnnealing
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from dataloader import TeluguTextDataset,create_telugu_dataloader
from tqdm import tqdm

from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig




dataset_registry: dict[str, Type[DataGen]] = {
    "form_language": FormLangDatasetGenerator
}

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_dataset(name, kwargs):
    cls = dataset_registry[name]
    return cls(from_dict(cls.config_class, OmegaConf.to_container(kwargs)))


def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.training.seed)

    # dataset = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)

    
    # dataset = TeluguTextDataset(cfg.dataset.data_dir, cfg.dataset.block_size)
    train_loader = create_telugu_dataloader(data_dir=cfg.dataset.data_dir, block_size=cfg.dataset.kwargs.context_length, batch_size=cfg.training.batch_size)
    # train_loader = DataLoader(dataset.train_split, batch_size=cfg.training.batch_size)
    # val_loaders = {
    #     key: DataLoader(val_ds, batch_size=cfg.training.batch_size) for key, val_ds in dataset.validation_split.items()
    # }
    # train_metrics = dataset.train_metrics.to(device=cfg.training.device)
    # val_metrics = dataset.validation_metrics.to(device=cfg.training.device)
    

    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
        device=cfg.training.device
    )
    model.reset_parameters()

    model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision],device=cfg.training.device)
    print("Number of Para : ", sum(p.numel() for p in model.parameters()))

    optim_groups = model._create_weight_decay_optim_groups()

    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=cfg.training.lr,
    )

    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        cfg.training.lr_warmup_steps,
        cfg.training.lr_decay_until_steps,
        cfg.training.lr,
        cfg.training.lr_decay_factor * cfg.training.lr,
    )

    # Training loop
    step = 0
    epoch = 1
    running_loss = 0.0
    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=0, initial=0)
        for inputs, labels in monitoring:
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)
            

            model.train()

            optimizer.zero_grad()
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):

                outputs = model(inputs.to(device=cfg.training.device))
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, cfg.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-1,
                )
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                running_loss = running_loss * step / (step + 1) + loss.item() * 1 / (step + 1)

            monitoring.set_description_str(f"Steps {step+1}/{cfg.training.num_steps} (Epoch: {epoch}) (Loss : {running_loss} )")

            step += 1
            if step % 300 == 0: # 19999
                PATH = os.path.join(r"E:\telugu_dataset_corpus\checkpoint", f"check_{step}_{running_loss}_.pth")
                model.eval()
                checkpoint = {
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler
                }
                torch.save(checkpoint, PATH)

            if step >= cfg.training.num_steps:
                break
        epoch += 1


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default="./parity_xlstm01")

    args = parser.parse_args()

    with open("./parity_xlstm01.yaml", "r", encoding="utf8") as fp:
        config_yaml = fp.read()

    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg)
