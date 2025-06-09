import os
import logging
from flask import Flask, render_template, request, jsonify
from model_service import TextReconstructionService

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "historical-text-reconstruction-key")

# Initialize the text reconstruction service
try:
    reconstruction_service = TextReconstructionService()
    logging.info("Successfully initialized text reconstruction service")
except Exception as e:
    logging.error(f"Failed to initialize reconstruction service: {e}")
    reconstruction_service = None

@app.route('/')
def index():
    """Main page with the text reconstruction interface."""
    return render_template('index.html')

@app.route('/reconstruct', methods=['POST'])
def reconstruct_text():
    """API endpoint for text reconstruction."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        damaged_text = data['text'].strip()
        if not damaged_text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if reconstruction_service is None:
            return jsonify({'error': 'Reconstruction model not available. Please check model files.'}), 500
        
        # Perform text reconstruction
        reconstructed_text = reconstruction_service.reconstruct(damaged_text)
        highlighted_text = reconstruction_service.highlight_insertions(damaged_text, reconstructed_text)
        
        return jsonify({
            'original': damaged_text,
            'reconstructed': reconstructed_text,
            'highlighted': highlighted_text,
            'success': True
        })
    
    except Exception as e:
        logging.error(f"Error during text reconstruction: {e}")
        return jsonify({'error': f'Reconstruction failed: {str(e)}'}), 500

@app.route('/examples')
def get_examples():
    """Get example texts for demonstration."""
    examples = [
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

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {error}")
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
