# Text Summarization AI

A comprehensive text summarization system using deep learning with both Basic LSTM and Attention mechanisms. This project provides an interactive web interface for text summarization with support for multiple file formats and model comparison.

## Features

### 🧠 Deep Learning Models
- **Basic LSTM Model**: Sequence-to-sequence model with LSTM layers
- **Attention Mechanism Model**: Advanced seq2seq model with attention for better context understanding
- **Model Comparison**: Side-by-side comparison of both models

### 🌐 Web Interface
- **Interactive UI**: Modern, responsive web interface
- **File Upload**: Support for .txt, .pdf, .doc, and .docx files
- **Drag & Drop**: Easy file upload with drag-and-drop functionality
- **Real-time Processing**: Live text processing and summarization

### 📊 Analysis Features
- **Text Statistics**: Word count, character count, and sentence analysis
- **Compression Ratio**: Measure of summarization effectiveness
- **Keyword Extraction**: Automatic keyword identification using TF-IDF
- **Keyword Highlighting**: Visual emphasis on important terms

### 🔧 Customization Options
- **Summary Length**: Choose between short (10%), medium (20%), or long (30%) summaries
- **Model Selection**: Switch between Basic LSTM and Attention models
- **Download Results**: Export summaries as text files

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended for training)

### Quick Start (Windows)
1. Double-click `run_windows.bat`
2. The script will automatically:
   - Create a virtual environment
   - Install all dependencies
   - Train the models (if not already trained)
   - Launch the web application

### Manual Installation
1. Clone or download the project
2. Navigate to the project directory
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **macOS/Linux**: `source venv/bin/activate`
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Application
1. Start the application:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`
3. Use the interface to:
   - Type or paste text directly
   - Upload files (.txt, .pdf, .doc, .docx)
   - Select model type and summary length
   - Generate summaries or compare models

### Command Line Tools

#### Training Models
```bash
# Train both models
python train.py

# Train specific model
python train.py --model basic
python train.py --model attention

# Custom training parameters
python train.py --epochs 10 --batch_size 32 --plot
```

#### Testing Models
```bash
# Test all functionality
python test_models.py

# Test specific functionality
python test_models.py single      # Test single model
python test_models.py compare     # Test model comparison
python test_models.py preprocessing # Test preprocessing
python test_models.py error       # Test error handling
```

#### Visualizations
```bash
# Create comprehensive visualizations
python visualize.py
```

## Project Structure

```
TextSum/
├── app.py                 # Flask web application
├── models.py              # Neural network models and preprocessing
├── train.py               # Training script
├── test_models.py         # Testing and evaluation script
├── visualize.py           # Visualization and analysis tools
├── config.json            # Configuration settings
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup script
├── run_windows.bat       # Windows launcher script
├── README.md             # This file
├── PROJECT_SUMMARY.md    # Project overview
├── QUICKSTART.md         # Quick start guide
├── templates/
│   └── index.html        # Web interface template
├── models/               # Saved model files (created after training)
├── uploads/              # Temporary file uploads
└── visualizations/       # Generated visualizations
```

## Model Architecture

### Basic LSTM Model
- **Encoder**: Embedding layer + LSTM layer
- **Decoder**: LSTM layer with encoder states as initial state
- **Output**: Dense layer with softmax activation
- **Training**: Teacher forcing with sparse categorical crossentropy

### Attention Model
- **Encoder**: Embedding layer + LSTM layer (returns sequences)
- **Decoder**: LSTM layer with attention mechanism
- **Attention**: Dot-product attention between encoder and decoder outputs
- **Output**: Dense layer with softmax activation

## Configuration

Edit `config.json` to customize:

```json
{
    "model_config": {
        "vocab_size": 10000,        # Vocabulary size
        "embedding_dim": 128,       # Embedding dimension
        "hidden_units": 256,        # LSTM hidden units
        "max_text_length": 500,     # Maximum input length
        "max_summary_length": 50,   # Maximum summary length
        "batch_size": 32,           # Training batch size
        "epochs": 10,               # Training epochs
        "learning_rate": 0.001      # Learning rate
    },
    "app_config": {
        "upload_folder": "uploads",
        "allowed_extensions": ["txt", "pdf", "doc", "docx"],
        "max_file_size": 16777216,  # 16MB
        "debug": true
    },
    "summary_lengths": {
        "short": 0.1,               # 10% of original length
        "medium": 0.2,              # 20% of original length
        "long": 0.3                 # 30% of original length
    }
}
```

## API Endpoints

### POST /api/summarize
Summarize text using a single model.

**Request Body:**
```json
{
    "text": "Text to summarize",
    "model_type": "basic" | "attention",
    "summary_length": "short" | "medium" | "long"
}
```

### POST /api/compare
Compare both models on the same text.

**Request Body:**
```json
{
    "text": "Text to summarize",
    "summary_length": "short" | "medium" | "long"
}
```

### POST /api/upload
Upload and process a file.

**Request:** Multipart form data with file

### POST /api/download
Download summary as text file.

**Request Body:**
```json
{
    "summary": "Summary text",
    "filename": "summary.txt"
}
```

## Training Data

The system uses a sample dataset of news articles with corresponding summaries. For production use, consider training on larger datasets such as:

- **CNN/DailyMail**: 300k article-summary pairs
- **XSum**: 230k BBC article-summary pairs
- **Multi-News**: 56k multi-document summaries
- **Custom Dataset**: Your domain-specific data

## Performance Optimization

### For Training
- Use GPU acceleration with CUDA-enabled TensorFlow
- Increase batch size for faster training
- Use mixed precision training for memory efficiency
- Implement data generators for large datasets

### For Inference
- Pre-load models at startup
- Implement model caching
- Use batch processing for multiple texts
- Consider model quantization for smaller models

## Troubleshooting

### Common Issues

1. **Models not found error**
   - Run `python train.py` to train models
   - Or use the Windows batch file `run_windows.bat`

2. **Memory errors during training**
   - Reduce batch size in config.json
   - Reduce max_text_length and max_summary_length
   - Close other applications to free memory

3. **File upload errors**
   - Check file size (max 16MB by default)
   - Ensure file format is supported
   - Verify file is not corrupted

4. **Poor summary quality**
   - Train for more epochs
   - Increase model complexity (hidden units)
   - Use larger training dataset
   - Adjust summary length settings

### Error Messages

- **"Text is too short for summarization"**: Input text must be at least 50 characters
- **"Models not trained"**: Run training script first
- **"File type not allowed"**: Use supported formats (.txt, .pdf, .doc, .docx)
- **"File is too large"**: Reduce file size or increase max_file_size in config

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Flask team for the web framework
- NLTK team for natural language processing tools
- Contributors to the CNN/DailyMail dataset

## Future Enhancements

- [ ] Support for more file formats (RTF, HTML, etc.)
- [ ] Multi-language summarization
- [ ] Extractive summarization option
- [ ] User authentication and saved summaries
- [ ] API rate limiting and caching
- [ ] Mobile app version
- [ ] Real-time collaborative summarization
- [ ] Integration with cloud storage services

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation wiki