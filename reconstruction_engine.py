import os
import logging
from dotenv import load_dotenv
load_dotenv()

class ReconstructionEngine:
    def __init__(self):
        self.model_path = "reconstruction_model"
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            if not os.path.exists(self.model_path):
                logging.warning(f"Model directory '{self.model_path}' not found. Using demo mode.")
                return
            required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.model_path, f))]
            if missing_files:
                logging.warning(f"Missing model files: {missing_files}. Using demo mode.")
                return
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logging.info("Model loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load model: {e}. Using demo mode.")

    def generate_reconstruction(self, damaged_text):
        if not self.model_loaded:
            return self._demo_reconstruction(damaged_text)
        try:
            import torch
            inputs = self.tokenizer(damaged_text, return_tensors="pt", truncation=True, padding=True)
            device = next(self.model.parameters()).device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            reconstructed = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return reconstructed.strip()
        except Exception as e:
            logging.error(f"Model inference failed: {e}")
            return self._demo_reconstruction(damaged_text)

    def _demo_reconstruction(self, damaged_text):
        reconstructions = {
            "Zаmfіrаkе. nꙋ'lꙋ꙼ аltъ": "Zamfirake. nulla alta",
            "Ѕъ nꙋ'lꙋ꙼ аltъ ԁаtъ": "Съ nulla alta data",
            "In nꙄmine P꙲tris et F꙲lii": "In nomine Patris et Filii",
            "Въ и͏̈мѧ ѻ҃ца и сн҃а": "Въ имя отца и сына"
        }
        if damaged_text in reconstructions:
            return reconstructions[damaged_text]
        return damaged_text

    def batch_reconstruct(self, texts):
        # Demo: Batch process a list of texts
        return [self.generate_reconstruction(t) for t in texts]

    def evaluate_bleu(self, predictions, references):
        # Demo BLEU calculation (not using external lib)
        # Returns average word overlap ratio
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            overlap = len(pred_words & ref_words)
            total = len(ref_words) if ref_words else 1
            scores.append(overlap / total)
        return sum(scores) / len(scores) if scores else 0.0

    def compute_exact_match(self, predictions, references):
        # Demo: Exact match accuracy
        matches = [p.strip() == r.strip() for p, r in zip(predictions, references)]
        return sum(matches) / len(matches) if matches else 0.0 