import os
import json
from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import pdfplumber
from models import SummarizationSystem
import tempfile
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

app.config['UPLOAD_FOLDER'] = config['app_config']['upload_folder']
app.config['MAX_CONTENT_LENGTH'] = config['app_config']['max_file_size']

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize summarization system
summarizer = SummarizationSystem()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config['app_config']['allowed_extensions']

def extract_text_from_file(file_path, file_extension):
    """Extract text from uploaded file"""
    try:
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        elif file_extension == 'pdf':
            text = ""
            try:
                # Try with pdfplumber first (better for complex PDFs)
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except:
                pass
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif file_extension in ['doc', 'docx']:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            return "Unsupported file format"
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """API endpoint for text summarization"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        model_type = data.get('model_type', 'basic')
        summary_length = data.get('summary_length', 'medium')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 50:
            return jsonify({'error': 'Text is too short for summarization (minimum 50 characters)'}), 400
        
        # Get length ratio from config
        length_ratio = config['summary_lengths'].get(summary_length, 0.2)
        
        # Generate summary
        summary = summarizer.summarize_text(text, model_type, length_ratio)
        
        # Get text statistics
        original_stats = summarizer.get_text_stats(text)
        summary_stats = summarizer.get_text_stats(summary)
        
        # Calculate compression ratio
        compression_ratio = summarizer.calculate_compression_ratio(text, summary)
        
        # Extract keywords
        keywords = summarizer.preprocessor.extract_keywords(summary, 5)
        
        # Highlight keywords in summary
        highlighted_summary = summarizer.highlight_keywords(summary, keywords)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'summary': summary,
            'highlighted_summary': highlighted_summary,
            'original_stats': original_stats,
            'summary_stats': summary_stats,
            'compression_ratio': compression_ratio,
            'keywords': keywords,
            'model_used': model_type,
            'summary_length': summary_length
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """API endpoint for comparing both models"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        summary_length = data.get('summary_length', 'medium')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 50:
            return jsonify({'error': 'Text is too short for summarization (minimum 50 characters)'}), 400
        
        # Get length ratio from config
        length_ratio = config['summary_lengths'].get(summary_length, 0.2)
        
        # Generate summaries with both models
        basic_summary = summarizer.summarize_text(text, 'basic', length_ratio)
        attention_summary = summarizer.summarize_text(text, 'attention', length_ratio)
        
        # Get statistics for both summaries
        basic_stats = summarizer.get_text_stats(basic_summary)
        attention_stats = summarizer.get_text_stats(attention_summary)
        
        # Calculate compression ratios
        basic_compression = summarizer.calculate_compression_ratio(text, basic_summary)
        attention_compression = summarizer.calculate_compression_ratio(text, attention_summary)
        
        # Extract keywords for both summaries
        basic_keywords = summarizer.preprocessor.extract_keywords(basic_summary, 5)
        attention_keywords = summarizer.preprocessor.extract_keywords(attention_summary, 5)
        
        # Highlight keywords
        basic_highlighted = summarizer.highlight_keywords(basic_summary, basic_keywords)
        attention_highlighted = summarizer.highlight_keywords(attention_summary, attention_keywords)
        
        # Get original text statistics
        original_stats = summarizer.get_text_stats(text)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'original_stats': original_stats,
            'basic_model': {
                'summary': basic_summary,
                'highlighted_summary': basic_highlighted,
                'stats': basic_stats,
                'compression_ratio': basic_compression,
                'keywords': basic_keywords
            },
            'attention_model': {
                'summary': attention_summary,
                'highlighted_summary': attention_highlighted,
                'stats': attention_stats,
                'compression_ratio': attention_compression,
                'keywords': attention_keywords
            },
            'summary_length': summary_length
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from file
            file_extension = filename.rsplit('.', 1)[1].lower()
            extracted_text = extract_text_from_file(file_path, file_extension)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            if extracted_text.startswith("Error"):
                return jsonify({'error': extracted_text}), 400
            
            return jsonify({
                'success': True,
                'text': extracted_text,
                'filename': filename
            })
        
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/download', methods=['POST'])
def download_summary():
    """Download summary as text file"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        summary = data.get('summary', '')
        filename = data.get('filename', 'summary.txt')
        
        if not summary:
            return jsonify({'error': 'No summary provided'}), 400
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        
        # Write summary with metadata
        content = f"Text Summary\n"
        content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"{'='*50}\n\n"
        content += summary
        
        temp_file.write(content)
        temp_file.close()
        
        return send_file(temp_file.name, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train the models (for development purposes)"""
    try:
        # This should be protected in production
        summarizer.train_models()
        return jsonify({'success': True, 'message': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        # Check if models are available
        models_available = os.path.exists('models/tokenizer.pkl')
        
        return jsonify({
            'models_available': models_available,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'allowed_extensions': config['app_config']['allowed_extensions'],
            'max_file_size': config['app_config']['max_file_size']
        })
    except Exception as e:
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if models exist, if not, train them
    if not os.path.exists('models/tokenizer.pkl'):
        print("Models not found. Training models...")
        try:
            summarizer.train_models()
            print("Models trained successfully!")
        except Exception as e:
            print(f"Error training models: {e}")
            print("You can train models manually by visiting /api/train")
    else:
        print("Loading existing models...")
        summarizer.load_models()
    
    app.run(debug=config['app_config']['debug'], host='0.0.0.0', port=5000)