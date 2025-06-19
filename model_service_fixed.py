import os
import logging
import re
from difflib import SequenceMatcher

class TextReconstructionService:
    """Service for handling text reconstruction using your trained model."""
    
    def __init__(self):
        """Initialize the reconstruction service."""
        self.model_path = "reconstruction_model"
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        
        # Initialize translation service
        self._init_translation()
        
        # Attempt to load your model
        self._load_model()
    
    def _init_translation(self):
        """Initialize the Gemini translation service."""
        self.translation_available = False
        try:
            import requests
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if self.api_key:
                self.translation_available = True
                logging.info("Gemini translation service initialized")
            else:
                logging.warning("GEMINI_API_KEY not found")
        except Exception as e:
            logging.error(f"Failed to initialize translation: {e}")
    
    def _load_model(self):
        """Load your trained model if files are present."""
        # Check if model directory exists
        if not os.path.exists(self.model_path):
            logging.warning(f"Model directory '{self.model_path}' not found. Using demo mode.")
            return
        
        # Check for required model files
        required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.model_path, f))]
        
        if missing_files:
            logging.warning(f"Missing model files: {missing_files}. Using demo mode.")
            return
        
        # Try to load dependencies and model
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logging.info(f"Loading your trained model from {self.model_path}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logging.info("Tokenizer loaded successfully")
            
            # Load model (automatically handles safetensors format)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logging.info("YOUR TRAINED MODEL LOADED SUCCESSFULLY!")
            
        except ImportError:
            logging.warning("PyTorch/transformers not installed. Using demo mode.")
        except Exception as e:
            logging.error(f"Failed to load your model: {e}. Using demo mode.")
    
    def reconstruct(self, damaged_text):
        """
        Reconstruct damaged text using your trained model.
        
        CHECKPOINT FLOW:
        1. Input processing (Latin/Cyrillic character handling)
        2. Model inference (your trained model)
        3. Output reconstruction
        """
        
        # CHECKPOINT 1: Input text processing - handles Latin/Cyrillic characters
        logging.info(f"CHECKPOINT 1 - Processing input: {damaged_text}")
        
        # If model not loaded, use demo
        if not self.model_loaded:
            logging.info("Using demo reconstruction - model not loaded")
            return self._demo_reconstruction(damaged_text)
        
        try:
            # CHECKPOINT 2: Your model inference
            logging.info("CHECKPOINT 2 - Using your trained model")
            
            # Your exact approach from notebook
            inputs = self.tokenizer(
                damaged_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            
            # Move inputs to model device
            import torch
            device = next(self.model.parameters()).device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Generate using your model
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            
            # Decode output
            reconstructed = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # CHECKPOINT 3: Output processing
            logging.info(f"CHECKPOINT 3 - Reconstructed: {reconstructed}")
            return reconstructed.strip()
            
        except Exception as e:
            logging.error(f"Model inference failed: {e}")
            return self._demo_reconstruction(damaged_text)
    
    def _demo_reconstruction(self, damaged_text):
        """Demo reconstruction when model not available."""
        # Your example mappings
        reconstructions = {
            "Zаmfіrаkе. nꙋ'lꙋ꙼ аltъ": "Zamfirake. nulla alta",
            "Ѕъ nꙋ'lꙋ꙼ аltъ ԁаtъ": "Съ nulla alta data",
            "In nꙄmine P꙲tris et F꙲lii": "In nomine Patris et Filii",
            "Въ и͏̈мѧ ѻ҃ца и сн҃а": "Въ имя отца и сына"
        }
        
        if damaged_text in reconstructions:
            return reconstructions[damaged_text]
        
        # Pattern-based replacements for demo
        result = damaged_text
        replacements = {
            'nꙄ': 'no', 'P꙲t': 'Pat', 'F꙲l': 'Fil', 'Sp꙲r': 'Spir', 'S꙲n': 'San',
            'ѻ҃ц': 'отец', 'сн҃': 'сын', 'ст҃': 'свят', 'д҃х': 'дух',
            'nꙋ\'l': 'null', 'аlt': 'alt', 'ԁаt': 'dat', 'Ѕъ': 'Съ'
        }
        
        for damaged, fixed in replacements.items():
            result = result.replace(damaged, fixed)
        
        return result
    
    def highlight_insertions(self, original, reconstructed):
        """Your exact highlighting logic using SequenceMatcher."""
        try:
            matcher = SequenceMatcher(None, original.split(), reconstructed.split())
            highlighted = []
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    highlighted.extend(reconstructed.split()[j1:j2])
                elif tag == "insert":
                    highlighted.extend([
                        f'<mark class="reconstruction">{word}</mark>' 
                        for word in reconstructed.split()[j1:j2]
                    ])
                elif tag == "replace":
                    highlighted.extend([
                        f'<mark class="reconstruction">{word}</mark>' 
                        for word in reconstructed.split()[j1:j2]
                    ])
            
            return ' '.join(highlighted)
            
        except Exception as e:
            logging.error(f"Highlighting error: {e}")
            return reconstructed
    
    def translate_with_gemini(self, text):
        """Translate using Gemini API with your exact prompt."""
        if not self.translation_available:
            return "Translation unavailable - Gemini API not configured"
        
        try:
            import requests
            import json
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
            
            # Your exact prompt from notebook
            prompt = f"""You are a Latin language expert. Translate the following Latin with Cyrillic letters into English and just give the final translation not all of the breakdown process:

Latin: {text}

English:"""
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    translation = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    return translation
                else:
                    return "Translation failed: No response from Gemini"
            else:
                logging.error(f"Gemini API error: {response.status_code}")
                return f"Translation failed: API error {response.status_code}"
                
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return f"Translation failed: {str(e)}"