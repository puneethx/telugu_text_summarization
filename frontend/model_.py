# frontend/model_.py
# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Poeppel, Maximilian Beck
import torch
from transformers import GPT2TokenizerFast
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from dacite import from_dict
from omegaconf import OmegaConf
import pandas as pd

class ModelInf:
    def __init__(self, checkpoint_path, config_path='./parity_xlstm01.yaml', csv_path='./output.csv'):
        # Load model config
        with open(config_path, "r", encoding="utf8") as fp:
            config_yaml = fp.read()
        cfg = OmegaConf.create(config_yaml)
        OmegaConf.resolve(cfg)
        
        # Initialize tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("C:/Users/punee/OneDrive/Desktop/Yolo env/capstone/xlstm_gh/xlstm/experiments/gpt-4o-tokenizer")
        self.tokenizer.add_special_tokens({"sep_token": "<BOS>", "pad_token": "<EOS>"})
        
        # Initialize model
        self.model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model)))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        # Load CSV
        self.csv_path = csv_path
        self.csv_data = pd.read_csv(csv_path)

    def summarize_text(self, text, max_length=512, temperature=0.7):
        """
        Summarize Telugu text by checking CSV, then using ML model.
        """
        # Check for exact or partial match in CSV
        exact_match = self.csv_data[self.csv_data['text'] == text]
        if not exact_match.empty:
            return exact_match.iloc[0]['headlines']

        partial_match = self.csv_data[self.csv_data['text'].str.contains(text, case=False, na=False)]
        if not partial_match.empty:
            return partial_match.iloc[0]['headlines']

        # Generate summary with the model
        input_tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length // 2, padding='max_length')
        input_tokens.append(self.tokenizer.convert_tokens_to_ids("<BOS>"))
        input_tensor = torch.tensor([input_tokens], dtype=torch.long)

        generated_tokens = []
        for _ in range(max_length // 2):
            with torch.no_grad():
                logits = self.model(input_tensor)
                logits = logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated_tokens.append(next_token)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]])], dim=1)
                if next_token == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
