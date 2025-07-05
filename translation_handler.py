import os
import logging
from dotenv import load_dotenv
load_dotenv()

class TranslationHandler:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.translation_available = bool(self.api_key)

    def translate_to_english(self, text):
        if not self.translation_available:
            return "Translation unavailable - API not configured"
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
            prompt = f"""You are a Latin language expert. Translate the following Latin with Cyrillic letters into English and just give the final translation not all of the breakdown process:\n\nLatin: {text}\n\nEnglish:"""
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    translation = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    return translation
                else:
                    return "Translation failed"
            else:
                logging.error(f"API error: {response.status_code}")
                return f"Translation failed: API error {response.status_code}"
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return f"Translation failed: {str(e)}"

    def format_translation(self, translation):
        # Add any formatting logic if needed
        return translation 