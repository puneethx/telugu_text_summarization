import torch
import os
from transformers import GPT2TokenizerFast
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from dacite import from_dict
from omegaconf import OmegaConf

def load_model(checkpoint_path, config_path='./parity_xlstm01.yaml'):
    """
    Load the trained xLSTM model and tokenizer
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        config_path (str): Path to the configuration YAML file
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Load configuration
    with open(config_path, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)

    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("./gpt-4o-tokenizer")
    special_tokens_dict = {
        "sep_token": "<BOS>",
        "pad_token": "<EOS>"
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Initialize model
    model = xLSTMLMModel(from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model)))
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, tokenizer

def generate_summary(model, tokenizer, text, max_length=512, temperature=0.7):
    """
    Generate a summary for the given text
    
    Args:
        model (xLSTMLMModel): Trained model
        tokenizer (GPT2TokenizerFast): Tokenizer
        text (str): Input text to summarize
        max_length (int): Maximum length of generated tokens
        temperature (float): Sampling temperature
    
    Returns:
        str: Generated summary
    """
    # Prepare input tokens
    input_tokens = tokenizer.encode(
        text, 
        max_length=max_length//2,
        truncation=True,
        padding='max_length',
        add_special_tokens=True
    )
    
    # Add separator token
    input_tokens.append(tokenizer.convert_tokens_to_ids("<BOS>"))
    
    # Prepare input tensor
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
    
    # Modify token ids to match model's expectations
    input_tensor = torch.where(input_tensor == 200020, 200001, input_tensor)
    input_tensor = torch.where(input_tensor == 200019, 200000, input_tensor)

    # Generation loop
    generated_tokens = []
    current_input = input_tensor
    
    with torch.no_grad():
        for _ in range(max_length//2):
            # Get model predictions
            outputs = model(current_input)
            
            # Sample from logits with temperature
            logits = outputs[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append token and update input
            generated_tokens.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            
            # Stop if end of sequence
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated tokens
    summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return summary

def main():
    # Example usage
    checkpoint_path = r"D:\Capstone\finetune_checkpoint\check_22350_2.2036365488818532_.pth"
    
    # Load model and tokenizer
    model, tokenizer = load_model(checkpoint_path)
    
    # Example text to summarize
    sample_text = "శనివారం ఇక్కడ జరిగిన రష్యన్ ఫెడరేషన్ కమ్యూనిస్టు పార్టీ 17వ జాతీయ మహాసభలకు అధ్యక్షుడు పుతిన్ అభినందనలు తెలియచేశారు. అక్టోబర్ మహావిప్లవ శతాబ్ది వేడుకలకు ఏర్పాట్లు జరుగుతున్న నేపథ్యంలో రాజధానిలోని రష్యన్ కమ్యూనిస్టు పార్టీ నిర్వహించిన ఈ వేదికపై అధ్యక్షుడు పుతిన్ పంపిన అభినందన సందేశాన్ని పార్టీ ప్రధాన కార్యదర్శి గ్రెనడీ జుగనోవ్ చదివి వినిపించారు. పుతిన్ అభినందనల సందేశాన్ని ఆయన ప్రస్తావిస్తూ దేశాన్ని మరింత బలోపేతం చేసేందుకు అధికారంలోవున్న అన్ని విభాగాలతో నిర్మాణాత్మక చర్చలు జరిపేందుకు తమ పార్టీ సిద్ధంగా వుందన్నారు. గత నాలుగేళ్ల కాలంలో తమ పార్టీలో 60 వేల మంది కొత్త సభ్యులు చేరారని ఆయన వివరించారు."
    
    # Generate summary
    summary = generate_summary(model, tokenizer, sample_text)
    print("Generated Summary:", summary)

if __name__ == "__main__":
    main()