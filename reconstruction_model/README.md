# Model Integration Guide for Historical Text Reconstruction

## Overview
This directory is where you should place your trained historical text reconstruction model files. The web application is already built and ready - you just need to integrate your specific model.

## Step 1: Install ML Dependencies

First, install the required packages. Run this command in your terminal:
```bash
pip install torch transformers
```

Or if using uv:
```bash
uv add torch transformers
```

## Step 2: Place Your Model Files

Place the following files from your trained model in this `reconstruction_model/` directory:

### Essential Model Files
- `config.json` - Model configuration file
- `model.safetensors` - The trained model weights (SafeTensors format)
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Tokenizer vocabulary and settings

### Additional Files (if applicable)
- `vocab.json` - Vocabulary file (for BPE models)
- `merges.txt` - Merge rules (for BPE models)
- `special_tokens_map.json` - Special tokens mapping
- `generation_config.json` - Generation configuration (optional)

## Step 3: Code Integration Points

### 1. Enable ML Imports (`model_service.py` - Lines 5-8)
Uncomment these lines at the top of the file:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

### 2. Enable Device Selection (`model_service.py` - Lines 18-19)
Uncomment this line in the `__init__` method:
```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 3. Model Loading (`model_service.py` - Lines 53-57)
Uncomment and modify these lines in the `_load_model()` method:
```python
logging.info(f"Loading model from {self.model_path}")
self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
self.model.to(self.device)
self.model.eval()
```

### 4. Model Inference (`model_service.py` - Lines 97-111)
Uncomment and modify the inference code in the `reconstruct()` method using your specific approach:
```python
inputs = self.tokenizer(
    damaged_text, 
    return_tensors="pt", 
    truncation=True, 
    padding=True
)

# Move inputs to the same device as model
device = next(self.model.parameters()).device
inputs = {key: val.to(device) for key, val in inputs.items()}

output = self.model.generate(**inputs, max_new_tokens=50)
reconstructed = self.tokenizer.decode(output[0], skip_special_tokens=True)

return reconstructed
```

## Step 4: Custom Model Integration

If your model uses a different architecture or loading method than HuggingFace transformers:

1. Replace the import statements with your model's requirements
2. Modify the model loading code to match your model's API
3. Update the inference logic to work with your model's input/output format
4. Adjust the highlighting logic if needed in the `highlight_insertions()` method

## Example from Your Notebook

Based on your example:
```python
damaged = "Ѕъ nꙋ'lꙋ꙼ аltъ ԁаtъ"
reconstructed = reconstruct(damaged)  # Your model's function
```

You would replace the inference section with your specific `reconstruct()` function call.

## Testing

1. The application currently runs in demo mode showing the UI
2. Once you integrate your model, the actual reconstruction will replace the demo responses
3. The highlighting system will automatically show which parts were reconstructed

## Current Status

- ✅ Web application with historical manuscript theme
- ✅ Light/dark mode toggle
- ✅ Responsive design with beautiful UI
- ✅ Example texts for testing
- ✅ Error handling and loading states
- ⏳ Model integration (your next step)

The application is fully functional and just waiting for your trained model integration!
