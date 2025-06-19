import os
import logging
import re
from difflib import SequenceMatcher

# NOTE: The following imports will be needed when you integrate your actual model
# Uncomment these when you have the model files ready:
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Translation service import - will be loaded conditionally
GEMINI_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
    logging.info("Using HTTP requests for Gemini API")
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available. Translation feature disabled.")

class TextReconstructionService:
    """Service for handling text reconstruction using the trained model."""
    
    def __init__(self):
        """Initialize the reconstruction service with the trained model."""
        self.model_path = "reconstruction_model"
        self.model = None
        self.tokenizer = None
        # Uncomment when integrating your model:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize translation service
        self._init_translation()
        
        # Load the model and tokenizer
        self._load_model()
    
    def _init_translation(self):
        """Initialize the Gemini translation service."""
        self.translation_available = False
        if REQUESTS_AVAILABLE:
            try:
                self.api_key = os.environ.get("GEMINI_API_KEY")
                if self.api_key:
                    self.translation_available = True
                    logging.info("Gemini translation service initialized successfully")
                else:
                    logging.warning("GEMINI_API_KEY not found. Translation feature disabled.")
            except Exception as e:
                logging.error(f"Failed to initialize Gemini translation: {e}")
                self.translation_available = False
    
    def _load_model(self):
        """Load the trained model and tokenizer from the model directory."""
        try:
            if not os.path.exists(self.model_path):
                logging.warning(f"Model directory '{self.model_path}' not found. Using demo mode.")
                self.model = "demo_mode"
                self.tokenizer = "demo_mode"
                return
            
            # Check for required model files
            required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.model_path, f))]
            
            if missing_files:
                logging.warning(f"Missing model files: {missing_files}. Using demo mode.")
                self.model = "demo_mode"
                self.tokenizer = "demo_mode"
                return
            
            # ============================================================================
            # MODEL INTEGRATION POINT: Replace this section with your trained model loading
            # ============================================================================
            # TODO: Uncomment and modify the following lines when you have your model ready:
            
            # First, uncomment the imports at the top of the file:
            # import torch
            # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Then uncomment and modify this loading code:
            # logging.info(f"Loading model from {self.model_path}")
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            # self.model.to(self.device)
            # self.model.eval()
            
            # If you're using a custom model architecture or different loading method,
            # replace the above lines with your specific implementation
            # ============================================================================
            
            # For now, use demo mode since dependencies aren't installed
            logging.info("Model files found but ML dependencies not installed. Using demo mode.")
            self.model = "demo_mode"
            self.tokenizer = "demo_mode"
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.model = "demo_mode"
            self.tokenizer = "demo_mode"
    
    def reconstruct(self, damaged_text):
        """
        Reconstruct the damaged text using the trained model.
        
        Args:
            damaged_text (str): The damaged/incomplete text to reconstruct
            
        Returns:
            str: The reconstructed complete text
        """
        try:
            if self.model == "demo_mode":
                # Demo reconstruction for testing the UI
                return self._demo_reconstruction(damaged_text)
            
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded properly")
            
            # ============================================================================
            # RECONSTRUCTION LOGIC: Your specific model inference code
            # ============================================================================
            # TODO: When you integrate your actual model, uncomment and modify this code:
            
            # inputs = self.tokenizer(
            #     damaged_text, 
            #     return_tensors="pt", 
            #     truncation=True, 
            #     padding=True
            # )
            
            # # Move inputs to the same device as model
            # device = next(self.model.parameters()).device
            # inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # output = self.model.generate(**inputs, max_new_tokens=50)
            # reconstructed = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # return reconstructed
            # ============================================================================
            
            # Placeholder return for when you integrate your model
            return damaged_text + " [Model integration needed]"
            
        except Exception as e:
            logging.error(f"Error during text reconstruction: {e}")
            return f"[Reconstruction Error: {str(e)}] Original: {damaged_text}"
    
    def _demo_reconstruction(self, damaged_text):
        """
        Demo reconstruction for testing purposes.
        This will be replaced when you integrate your actual model.
        """
        # Demo reconstructions based on your example
        reconstructions = {
            "Ѕъ nꙋ'lꙋ꙼ аltъ ԁаtъ": "Съ nulla alta data",
            "Zаmfіrаkе. nꙋ'lꙋ꙼ аltъ": "Zamfirake. nulla alta",
            "In nꙄmine P꙲tris et F꙲lii": "In nomine Patris et Filii",
            "Въ и͏̈мѧ ѻ҃ца и сн҃а": "Въ имя отца и сына"
        }
        
        # Check for exact matches first
        if damaged_text in reconstructions:
            return reconstructions[damaged_text]
        
        # Simple pattern-based reconstruction for demo
        result = damaged_text
        
        # Replace common abbreviations and damaged characters
        replacements = {
            'nꙄ': 'no',
            'P꙲t': 'Pat',
            'F꙲l': 'Fil',
            'Sp꙲r': 'Spir',
            'S꙲n': 'San',
            'ѻ҃ц': 'отец',
            'сн҃': 'сын',
            'ст҃': 'свят',
            'д҃х': 'дух',
            'nꙋ\'l': 'null',
            'аlt': 'alt',
            'ԁаt': 'dat'
        }
        
        for damaged, fixed in replacements.items():
            result = result.replace(damaged, fixed)
        
        return result
    
    def highlight_insertions(self, original, reconstructed):
        """
        Highlight the insertions/reconstructions in the text using SequenceMatcher.
        
        Args:
            original (str): The original damaged text
            reconstructed (str): The reconstructed text
            
        Returns:
            str: HTML string with highlighted insertions
        """
        try:
            # ============================================================================
            # HIGHLIGHTING LOGIC: Using your specific SequenceMatcher approach
            # ============================================================================
            matcher = SequenceMatcher(None, original.split(), reconstructed.split())
            highlighted = []
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    # Words that are the same in both texts
                    highlighted.extend(reconstructed.split()[j1:j2])
                elif tag == "insert":
                    # Words that were inserted/reconstructed
                    highlighted.extend([
                        f'<mark class="reconstruction">{word}</mark>' 
                        for word in reconstructed.split()[j1:j2]
                    ])
                elif tag == "replace":
                    # Words that were replaced/reconstructed
                    highlighted.extend([
                        f'<mark class="reconstruction">{word}</mark>' 
                        for word in reconstructed.split()[j1:j2]
                    ])
            
            return ' '.join(highlighted)
            
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
    
    def translate_with_gemini(self, text):
        """
        Translate reconstructed Latin/Cyrillic text to English using Gemini API.
        
        Args:
            text (str): The reconstructed text to translate
            
        Returns:
            str: English translation or error message
        """
        if not self.translation_available:
            return "Translation unavailable - Gemini API not configured"
        
        try:
            import requests
            import json
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
            
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
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    translation = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    return translation
                else:
                    return "Translation failed: No response from Gemini"
            else:
                logging.error(f"Gemini API error: {response.status_code} - {response.text}")
                return f"Translation failed: API error {response.status_code}"
            
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return f"Translation failed: {str(e)}"
