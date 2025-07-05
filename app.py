import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
from input_handler import InputHandler
from reconstruction_engine import ReconstructionEngine
from translation_handler import TranslationHandler
from output_handler import OutputHandler
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "historical-text-reconstruction-key")

# Initialize handlers/services
input_handler = InputHandler()
reconstruction_engine = ReconstructionEngine()
translation_handler = TranslationHandler()
output_handler = OutputHandler()

@app.route('/')
def index():
    """Main page with the text reconstruction interface."""
    return render_template('index.html')

@app.route('/reconstruct', methods=['POST'])
def reconstruct_text():
    """API endpoint for text reconstruction using modular handlers."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        # Input validation
        valid, cleaned = input_handler.validate_text(data['text'])
        if not valid:
            return jsonify({'error': cleaned}), 400
        # Preprocess input
        preprocessed = input_handler.preprocess_input(cleaned)
        # Generate reconstruction
        reconstructed = reconstruction_engine.generate_reconstruction(preprocessed)
        # Highlight insertions
        highlighted = output_handler.highlight_insertions(preprocessed, reconstructed)
        # Translate
        translation = translation_handler.translate_to_english(reconstructed)
        formatted_translation = translation_handler.format_translation(translation)
        # Display results
        result = output_handler.display_results(preprocessed, reconstructed, highlighted, formatted_translation)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during text reconstruction: {e}")
        return jsonify({'error': f'Reconstruction failed: {str(e)}'}), 500

@app.route('/examples')
def get_examples():
    """Get example texts for demonstration."""
    examples = [
        {
            'title': 'Your Example Text',
            'damaged': 'Zаmfіrаkе. nꙋ\'lꙋ꙼ аltъ',
            'description': 'Example from your notebook showing mixed Latin/Cyrillic'
        },
        {
            'title': 'Latin Medieval Text',
            'damaged': 'In nꙄmine P꙲tris et F꙲lii et Sp꙲ritus S꙲ncti',
            'description': 'Medieval Latin religious text with abbreviated words'
        },
        {
            'title': 'Cyrillic Church Slavonic',
            'damaged': 'Въ и͏̈мѧ ѻ҃ца и сн҃а и ст҃аго д҃ха',
            'description': 'Church Slavonic liturgical text with missing characters'
        },
        {
            'title': 'Mixed Script Historical',
            'damaged': 'Ѕъ nꙋ\'lꙋ꙼ аltъ ԁаtъ',
            'description': 'Historical text mixing Latin and Cyrillic scripts'
        }
    ]
    return jsonify(examples)

@app.route('/download', methods=['POST'])
def download_file():
    data = request.get_json()
    original = data.get('original', '')
    reconstructed = data.get('reconstructed', '')
    translation = data.get('translation', '')
    file_format = data.get('format', 'txt')

    if file_format != 'txt':
        return jsonify({'error': 'Only .txt download is supported.'}), 400

    txt = (
        'Historical Text Reconstruction\n\n'
        f'Original Damaged Text:\n{original}\n\n'
        f'Reconstructed Text:\n{reconstructed}\n\n'
        f'English Translation:\n{translation}\n'
    )
    txt_output = BytesIO(txt.encode('utf-8'))
    return send_file(txt_output, as_attachment=True, download_name='reconstruction.txt', mimetype='text/plain')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {error}")
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
