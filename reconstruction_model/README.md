# Model Integration Instructions

## Overview
This directory is where you should place your trained historical text reconstruction model files.

## Required Files
Place the following files from your trained model in this directory:

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

## Integration Points in Code

### 1. Model Loading (`model_service.py`)
The main integration point is in the `_load_model()` method of the `TextReconstructionService` class:

```python
# Line ~35-50 in model_service.py
# Replace this section with your specific model loading code
self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
