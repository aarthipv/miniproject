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
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize_input(self, text):
        return text.split()

    def count_characters(self, text):
        return len(text)

    def count_words(self, text):
        return len(self.tokenize_input(text))

    