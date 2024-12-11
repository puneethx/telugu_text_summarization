import os
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import List, Tuple
import glob
from transformers import GPT2TokenizerFast

class TeluguTextDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 block_size: int = 128):
        self.block_size = block_size
        self.tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-4o-tokenizer")  #encoding_for_model
        # print("Vocab size : ", self.tokenizer.vocab)
        self.file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
        
        self.tokenized_chunks = []
        
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                tokens = self.tokenizer.encode(text)
                
                for i in range(0, len(tokens) - block_size, block_size):
                    chunk = tokens[i:i + block_size + 1]
                    if len(chunk) == block_size + 1:
                        self.tokenized_chunks.append(
                            chunk #don't convert to tensor now 
                        )
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
            print(file_path)
            
    
    def __len__(self) -> int:
        return len(self.tokenized_chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokenized_chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # Input sequence 
        y = torch.tensor(chunk[1:], dtype=torch.long)   # Target sequence
        return x, y

def create_telugu_dataloader(
    data_dir: str,
    block_size: int = 64,
    batch_size: int = 4,
    shuffle: bool = True,
    # num_workers: int = 4
) -> DataLoader:
    dataset = TeluguTextDataset(data_dir, block_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=num_workers,
        pin_memory=True
    )