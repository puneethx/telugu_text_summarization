import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import GPT2TokenizerFast

class TeluguSummarizationDataset(Dataset):
    """
    Dataset class for Telugu text summarization
    """
    def __init__(self, csv_path, tokenizer, max_length=512):
        """
        Args:
            csv_path (str): Path to the CSV file containing data
            tokenizer (GPT2TokenizerFast): Tokenizer for encoding text
            max_length (int): Maximum token sequence length
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Add special tokens to tokenizer
        special_tokens_dict = {
            "sep_token": "<BOS>",
            "pad_token": "<EOS>"
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        self.sep_token = "<BOS>"
        self.pad_token = "<EOS>"

        # Fill missing data with empty strings
        self.data['text'] = self.data['text'].fillna('')
        self.data['headlines'] = self.data['headlines'].fillna('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches the item at index `idx`.
        """
        try:
            # Extract text and headline
            text = str(self.data.iloc[idx]['text'])
            headline = str(self.data.iloc[idx]['headlines'])

            # Tokenize text and headline
            text_tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length // 2,
                truncation=True,
                padding='max_length',
                add_special_tokens=True
            )
            headline_tokens = self.tokenizer.encode(
                headline,
                max_length=self.max_length // 2,
                truncation=True,
                padding='max_length',
                add_special_tokens=True
            )

            # Combine tokens with special separator
            combined_input = (
                text_tokens +
                [self.tokenizer.convert_tokens_to_ids(self.sep_token)] +
                headline_tokens
            )

            # Truncate if longer than max_length
            if len(combined_input) > self.max_length:
                combined_input = combined_input[:self.max_length + 1]

            # Replace special token IDs as required
            combined_input = [200001 if x == 200020 else x for x in combined_input]
            combined_input = [200000 if x == 200019 else x for x in combined_input]

            # Create input and target tensors
            input_tensor = torch.tensor(combined_input[:-1], dtype=torch.long)
            target_tensor = torch.tensor(combined_input[1:], dtype=torch.long)

            return input_tensor, target_tensor

        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            # Return dummy tensors on error
            dummy = torch.ones(self.max_length - 1, dtype=torch.long) * self.tokenizer.pad_token_id
            return dummy, dummy


def create_summarization_dataloader(csv_path, tokenizer, batch_size, max_length=512):
    """
    Creates a DataLoader for the summarization dataset.
    
    Args:
        csv_path (str): Path to the CSV file
        tokenizer (GPT2TokenizerFast): Tokenizer for text encoding
        batch_size (int): Batch size for the DataLoader
        max_length (int): Maximum sequence length
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = TeluguSummarizationDataset(csv_path, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to >0 for multiprocessing during data loading
        pin_memory=True
    )


# Example usage
if __name__ == "__main__":
    csv_path = r"C:\Users\punee\OneDrive\Desktop\Yolo env\capstone\jsontocsv\output.csv"
    tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-4o-tokenizer")

    batch_size = 4
    max_length = 512

    # Instantiate the dataset
    dataset = TeluguSummarizationDataset(csv_path, tokenizer, max_length)

    # Print first four examples of text and headlines
    print("First four examples from the dataset:")
    for i in range(min(4, len(dataset))):
        text = dataset.data.iloc[i]['text']
        headline = dataset.data.iloc[i]['headlines']
        print(f"Example {i + 1}:")
        print("Text:", text)
        print("headline:", headline)
        print("-" * 50)

    # Load DataLoader
    dataloader = create_summarization_dataloader(csv_path, tokenizer, batch_size, max_length)

    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print("Inputs:", inputs)
        print("Targets:", targets)
        if i == 2:  # Limit to first 3 batches for demo
            break
