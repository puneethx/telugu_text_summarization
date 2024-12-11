# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
from argparse import ArgumentParser
from typing import Type
import os
import torch
import torch.optim as optim
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
import pandas as pd
from transformers import GPT2TokenizerFast

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

class TeluguSummarizationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        special_tokens_dict = {
            "sep_token" : "<BOS>",
            "pad_token" : "<EOS>"
        }

        tokenizer.add_special_tokens(special_tokens_dict)
        self.sep_token = "<BOS>"
        self.pad_token = "<EOS>"
        
        # Clean the data
        self.data['text'] = self.data['text'].fillna('')
        self.data['headlines'] = self.data['headlines'].fillna('')
        # print(self.data.head)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            text = str(self.data.iloc[idx]['text'])
            headline = str(self.data.iloc[idx]['headlines'])
            

            # # Tokenize with proper handling of special tokens
            text_tokens = self.tokenizer.encode(
                text, 
                max_length=self.max_length//2,
                truncation=True,
                padding='max_length',
                add_special_tokens=True
            )
            
            headline_tokens = self.tokenizer.encode(
                headline,
                max_length=self.max_length//2,
                truncation=True,
                padding='max_length',
                add_special_tokens=True
            )
            
            combined_input = (
                text_tokens + 
                [self.tokenizer.convert_tokens_to_ids(self.sep_token)] + 
                headline_tokens
            )
            
            if len(combined_input) > self.max_length:
                combined_input = combined_input[:self.max_length+1]
            combined_input = [200001 if x == 200020 else x for x in combined_input]
            combined_input = [200000 if x == 200019 else x for x in combined_input]
            # print(combined_input)
            # # Create input and target tensors
            input_tensor = torch.tensor(combined_input[:-1], dtype=torch.long)
            target_tensor = torch.tensor(combined_input[1:], dtype=torch.long)
            
            return input_tensor, target_tensor
        
            # return text_tokens, headline_tokens
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            # Return a dummy tensor pair in case of error
            dummy = torch.ones(self.max_length-1, dtype=torch.long) * self.tokenizer.pad_token_id
            return dummy, dummy

def create_summarization_dataloader(csv_path, tokenizer, batch_size, max_length=512):
    dataset = TeluguSummarizationDataset(csv_path, tokenizer, max_length)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True
    )


def fine_tune_xlstm(pretrained_path, csv_path, output_dir, cfg):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.training.seed)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-4o-tokenizer")

    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))).to(
        device=cfg.training.device
    )

    model.reset_parameters() ##commented when loaded from existing model

    checkpoint = torch.load(pretrained_path, map_location=cfg.training.device)


    model.load_state_dict(checkpoint['model'])

    # model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision], device=cfg.training.device)

    train_loader = create_summarization_dataloader(
        csv_path,
        tokenizer,
        cfg.training.batch_size,
        cfg.dataset.kwargs.context_length
    )

    print("Number of Param : ", sum(p.numel() for p in model.parameters()))

    optim_groups = model._create_weight_decay_optim_groups()

    optimizer = optim.AdamW(
        (
            {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
        lr=cfg.training.lr,
    )

    step = 0
    epoch = 1
    running_loss = 0.0

    while step < cfg.training.num_steps:
        monitoring = tqdm(train_loader, total=0, initial=0)
        model.train()
        for inputs, labels in monitoring:
            inputs = inputs.to(device=cfg.training.device)
            labels = labels.to(device=cfg.training.device)


            optimizer.zero_grad()
            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch_dtype_map[cfg.training.amp_precision],
                enabled=cfg.training.enable_mixed_precision,
            ):
                # print(inputs.shape, labels.shape)
                outputs = model(inputs)
                # print(outputs.shape)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, cfg.model.vocab_size),
                    labels.view(-1),
                    ignore_index=-1,
                )
                loss.backward()
                optimizer.step()
                running_loss = running_loss * step / (step + 1) + loss.item() * 1 / (step + 1)

            monitoring.set_description_str(f"Steps {step+1}/{cfg.training.num_steps} (Epoch: {epoch}) (Loss : {running_loss} )")

            step += 1
            if step % 50 == 0: # 19999
                PATH = os.path.join(output_dir, f"check_{step}_{running_loss}_.pth")
                model.eval()
                checkpoint = {
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, PATH)

            if step >= cfg.training.num_steps:
                break
        epoch += 1

if __name__ == "__main__":
    with open("./parity_xlstm01.yaml", "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    
    # cfg.training.num_steps = 50000
    # cfg.training.batch_size = 4
    
    fine_tune_xlstm(
        pretrained_path=r"D:\Capstone\checkpoint\check_3000_3.965958630641282_.pth",
        csv_path=r"C:\Users\punee\OneDrive\Desktop\Yolo env\capstone\jsontocsv\output.csv",
        output_dir=r"D:\Capstone\checkpoint\finetune",
        cfg=cfg
    )
