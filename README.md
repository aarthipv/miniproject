# Historical Text Reconstruction Web App

Restore damaged historical Latin and Cyrillic texts using advanced machine learning and a beautiful web interface.

## Features
- Reconstructs damaged manuscript text (Latin/Cyrillic)
- Highlights reconstructed portions
- Provides English translation (if Gemini API key is set)
- Download results as a .txt file
- Example texts for easy testing
- Light/dark manuscript-themed UI

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd miniproject
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **(Optional) Set up Gemini API key for translation:**
   - Create a `.env` file in the project root:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

4. **(Optional) Add your trained model:**
   - Place your model files (`config.json`, `model.safetensors`, `tokenizer_config.json`, `tokenizer.json`) in the `reconstruction_model/` directory.
   - If no model is present, the app runs in demo mode.

## Running the App

```sh
python app.py
```

- The app will be available at [http://127.0.0.1:5002](http://127.0.0.1:5002)

## Usage
1. Enter damaged text in the input box.
2. Click "Reconstruct Text" to see the reconstruction, highlights, and translation.
3. Use the "Download as .txt" button to save the results.
4. Try example texts from the dropdown for quick demos.

## Project Structure
```
miniproject/
├── app.py
├── input_handler.py
├── reconstruction_engine.py
├── translation_handler.py
├── output_handler.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── reconstruction_model/  # Place your model files here
└── ...
```

## Notes
- If you want to use your own model, see `reconstruction_model/README.md` for integration instructions.
- The app will run in demo mode if no model is present.
- Only .txt download is supported (PDF download is not available).

## License
MIT 