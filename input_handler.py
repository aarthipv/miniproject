import re

class InputHandler:
    def validate_text(self, text):
        if not text or not isinstance(text, str):
            return False, 'No text provided.'
        cleaned = text.strip()
        if not cleaned:
            return False, 'Empty text provided.'
        return True, cleaned

    def preprocess_input(self, text):
        # Normalize whitespace and remove unwanted characters
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize_input(self, text):
        # Simple whitespace tokenizer (demo)
        return text.split()

    def count_characters(self, text):
        # Return character count (demo)
        return len(text)

    def count_words(self, text):
        # Return word count (demo)
        return len(self.tokenize_input(text))

    def detect_script(self, text):
        # Demo: Detect if text contains Latin, Cyrillic, or both
        latin = bool(re.search(r'[A-Za-z]', text))
        cyrillic = bool(re.search(r'[\u0400-\u04FF]', text))
        if latin and cyrillic:
            return 'Mixed'
        elif latin:
            return 'Latin'
        elif cyrillic:
            return 'Cyrillic'
        else:
            return 'Unknown' 