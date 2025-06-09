import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

class TextReconstructionService:
    """Service for handling text reconstruction using the trained model."""
    
    def __init__(self):
        """Initialize the reconstruction service with the trained model."""
        self.model_path = "reconstruction_model"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer from the model directory."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory '{self.model_path}' not found. Please place your trained model files in this directory.")
            
            # Check for required model files
            required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.model_path, f))]
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {missing_files}")
            
            # ============================================================================
            # MODEL INTEGRATION POINT: Replace this section with your trained model loading
            # ============================================================================
            # TODO: Replace the following lines with your specific model loading code
            # Example for HuggingFace transformers:
            
            logging.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # If you're using a custom model architecture or different loading method,
            # replace the above lines with your specific implementation
            # ============================================================================
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise e
    
    def reconstruct(self, damaged_text):
        """
        Reconstruct the damaged text using the trained model.
        
        Args:
            damaged_text (str): The damaged/incomplete text to reconstruct
            
        Returns:
            str: The reconstructed complete text
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded properly")
            
            # ============================================================================
            # RECONSTRUCTION LOGIC: Replace this section with your model's inference code
            # ============================================================================
            # TODO: Replace the following logic with your specific model inference
            
            # Tokenize the input
            inputs = self.tokenizer(
                damaged_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate reconstruction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode the output
            reconstructed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # If your model expects different input format or generates different output,
            # modify the above code accordingly
            # ============================================================================
            
            return reconstructed.strip()
            
        except Exception as e:
            logging.error(f"Error during text reconstruction: {e}")
            # Return a fallback response for demonstration
            return f"[Reconstruction Error: {str(e)}] Original: {damaged_text}"
    
    def highlight_insertions(self, original, reconstructed):
        """
        Highlight the insertions/reconstructions in the text.
        
        Args:
            original (str): The original damaged text
            reconstructed (str): The reconstructed text
            
        Returns:
            str: HTML string with highlighted insertions
        """
        try:
            # ============================================================================
            # HIGHLIGHTING LOGIC: Customize this section based on your needs
            # ============================================================================
            # This is a simple implementation - you may want to use more sophisticated
            # text alignment algorithms for better highlighting
            
            # Split texts into words for comparison
            orig_words = original.split()
            recon_words = reconstructed.split()
            
            highlighted_parts = []
            orig_idx = 0
            
            for recon_word in recon_words:
                if orig_idx < len(orig_words) and self._words_similar(orig_words[orig_idx], recon_word):
                    # Word exists in original, keep as is
                    highlighted_parts.append(recon_word)
                    orig_idx += 1
                else:
                    # This is likely an insertion/reconstruction
                    highlighted_parts.append(f'<mark class="reconstruction">{recon_word}</mark>')
            
            return ' '.join(highlighted_parts)
            
        except Exception as e:
            logging.error(f"Error during text highlighting: {e}")
            return reconstructed
    
    def _words_similar(self, word1, word2, threshold=0.8):
        """
        Check if two words are similar (allowing for minor differences).
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            threshold (float): Similarity threshold
            
        Returns:
            bool: True if words are similar enough
        """
        # Simple similarity check - you can implement more sophisticated methods
        if word1 == word2:
            return True
        
        # Remove common historical text markers and compare
        clean_word1 = re.sub(r'[꙼꙲Ꙅ]*', '', word1.lower())
        clean_word2 = re.sub(r'[꙼꙲Ꙅ]*', '', word2.lower())
        
        if clean_word1 == clean_word2:
            return True
        
        # Check if one word contains the other (for abbreviations)
        if clean_word1 in clean_word2 or clean_word2 in clean_word1:
            return len(max(clean_word1, clean_word2, key=len)) / len(min(clean_word1, clean_word2, key=len)) <= 2
        
        return False
