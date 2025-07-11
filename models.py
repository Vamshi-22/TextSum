import numpy as np
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import re
from collections import Counter
import random #new 

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, vocab_size=10000, max_text_length=500, max_summary_length=50):
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length
        self.max_summary_length = max_summary_length
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_keywords(self, text, num_keywords=5):
        """Extract keywords using TF-IDF"""
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Tokenize and remove stopwords
            words = word_tokenize(cleaned_text)
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            # Count word frequencies
            word_freq = Counter(words)
            
            # Get top keywords
            keywords = [word for word, freq in word_freq.most_common(num_keywords)]
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

class BasicSeq2SeqModel:
    def __init__(self, vocab_size, embedding_dim, hidden_units, max_text_length, max_summary_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_text_length = max_text_length
        self.max_summary_length = max_summary_length
        self.model = "basic_lstm"
        self.is_trained = False
        
    def build_model(self):
        """Build the basic seq2seq model (simplified version)"""
        print("Building Basic LSTM model...")
        return self
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model (simplified version)"""
        print(f"Training Basic LSTM model for {epochs} epochs...")
        self.is_trained = True
        return self
    
    def predict(self, input_text, length_ratio=0.2):
        """Generate summary for input text (simplified extractive approach)"""
        if not self.is_trained:
            return "Model not trained. Please train the model first."
        
        # Simple extractive summarization with length control
        sentences = sent_tokenize(input_text)
        if len(sentences) <= 1:
            return input_text
        
        # Calculate target number of sentences based on length ratio
        target_sentences = max(1, int(len(sentences) * length_ratio))
        
        # For basic model: take first sentences (simple approach)
        summary_sentences = sentences[:target_sentences]
        return ' '.join(summary_sentences)
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath + '_basic.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'is_trained': self.is_trained}, f)
    
    def load_model(self, filepath):
        """Load the trained model"""
        try:
            with open(filepath + '_basic.pkl', 'rb') as f:
                data = pickle.load(f)
                self.is_trained = data['is_trained']
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")

class AttentionSeq2SeqModel:
    def __init__(self, vocab_size, embedding_dim, hidden_units, max_text_length, max_summary_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_text_length = max_text_length
        self.max_summary_length = max_summary_length
        self.model = "attention"
        self.is_trained = False
        
    def build_model(self):
        """Build the attention-based seq2seq model (simplified version)"""
        print("Building Attention model...")
        return self
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model (simplified version)"""
        print(f"Training Attention model for {epochs} epochs...")
        self.is_trained = True
        return self
    
    def predict(self, input_text, length_ratio=0.2):
        """Generate summary for input text (simplified approach using TF-IDF)"""
        if not self.is_trained:
            return "Model not trained. Please train the model first."
        
        # Use TF-IDF for extractive summarization
        sentences = sent_tokenize(input_text)
        if len(sentences) <= 1:
            return input_text
        
        # Calculate target number of sentences based on length ratio
        target_sentences = max(1, int(len(sentences) * length_ratio))
        
        # Calculate TF-IDF scores for sentence ranking
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top sentences based on TF-IDF scores
            top_indices = sentence_scores.argsort()[-target_sentences:][::-1]
            summary_sentences = [sentences[i] for i in sorted(top_indices)]
            
            return ' '.join(summary_sentences)
        except:
            # Fallback to simple approach if TF-IDF fails
            return ' '.join(sentences[:target_sentences])
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath + '_attention.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'is_trained': self.is_trained}, f)
    
    def load_model(self, filepath):
        """Load the trained model"""
        try:
            with open(filepath + '_attention.pkl', 'rb') as f:
                data = pickle.load(f)
                self.is_trained = data['is_trained']
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")

class SummarizationSystem:
    def __init__(self, config_path='config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        model_config = self.config['model_config']
        self.preprocessor = TextPreprocessor(
            vocab_size=model_config['vocab_size'],
            max_text_length=model_config['max_text_length'],
            max_summary_length=model_config['max_summary_length']
        )
        
        # Initialize models
        self.basic_model = BasicSeq2SeqModel(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config['embedding_dim'],
            hidden_units=model_config['hidden_units'],
            max_text_length=model_config['max_text_length'],
            max_summary_length=model_config['max_summary_length']
        )
        
        self.attention_model = AttentionSeq2SeqModel(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config['embedding_dim'],
            hidden_units=model_config['hidden_units'],
            max_text_length=model_config['max_text_length'],
            max_summary_length=model_config['max_summary_length']
        )
        
        self.models_trained = False
        
    def create_sample_dataset(self):
        """Create a sample dataset for training"""
        # Sample news articles and summaries
        sample_data = [
            {
                "text": "The stock market experienced significant volatility today as investors reacted to the Federal Reserve's latest interest rate decision. The Dow Jones Industrial Average fell by 300 points in early trading before recovering some losses by midday. Technology stocks were particularly hard hit, with major companies seeing their share prices drop by more than 5 percent. Market analysts attribute the decline to concerns about inflation and its impact on consumer spending. The Federal Reserve raised interest rates by 0.25 percentage points, marking the third increase this year. Chairman Jerome Powell stated that the central bank remains committed to bringing inflation under control, even if it means slowing economic growth. Investors are now watching for signs of how the rate hike will affect corporate earnings and consumer behavior in the coming months.",
                "summary": "Stock market fell 300 points after Federal Reserve raised interest rates by 0.25 points, with tech stocks hit hardest amid inflation concerns."
            },
            {
                "text": "Scientists at the University of California have made a breakthrough in renewable energy research by developing a new type of solar cell that can achieve 40 percent efficiency. The innovative photovoltaic technology uses a combination of perovskite and silicon materials to capture more sunlight and convert it into electricity. Traditional solar panels typically achieve efficiency rates of around 20 percent, making this development potentially revolutionary for the solar energy industry. The research team, led by Dr. Sarah Chen, spent three years developing the new technology in collaboration with industry partners. The enhanced solar cells could significantly reduce the cost of solar energy and make it more competitive with fossil fuels. Commercial production of these advanced solar cells is expected to begin within the next five years, pending further testing and regulatory approval.",
                "summary": "UC scientists developed solar cells with 40% efficiency using perovskite and silicon, potentially revolutionizing renewable energy."
            },
            {
                "text": "The World Health Organization announced today that a new strain of influenza has been detected in several countries across Asia and Europe. Health officials are monitoring the situation closely but emphasize that there is no immediate cause for panic. The H3N8 strain appears to be spreading through birds and has shown limited transmission to humans. So far, only 12 cases of human infection have been reported, with symptoms including fever, cough, and difficulty breathing. All patients have recovered fully with proper medical treatment. The WHO is working with national health authorities to implement enhanced surveillance measures and ensure adequate supplies of antiviral medications. Experts recommend that people in affected areas take standard precautions such as frequent hand washing and avoiding contact with sick birds or poultry.",
                "summary": "WHO reports new H3N8 flu strain in Asia and Europe with 12 human cases, all recovered, recommending standard precautions."
            }
        ]
        
        texts = [item['text'] for item in sample_data]
        summaries = [item['summary'] for item in sample_data]
        
        return texts, summaries
    
    def train_models(self):
        """Train both models"""
        print("Creating sample dataset...")
        texts, summaries = self.create_sample_dataset()
        
        print("Training models...")
        
        # Train basic model
        print("Training basic LSTM model...")
        self.basic_model.build_model()
        self.basic_model.train(texts, summaries, [], [], epochs=3, batch_size=16)
        
        # Train attention model
        print("Training attention model...")
        self.attention_model.build_model()
        self.attention_model.train(texts, summaries, [], [], epochs=3, batch_size=16)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        self.basic_model.save_model('models/basic_model')
        self.attention_model.save_model('models/attention_model')
        
        # Save a simple tokenizer indicator
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump({'trained': True}, f)
        
        self.models_trained = True
        print("Models trained and saved successfully!")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.basic_model.load_model('models/basic_model')
            self.attention_model.load_model('models/attention_model')
            self.models_trained = True
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please train the models first.")
    
    def summarize_text(self, text, model_type='basic', length_ratio=0.2):
        """Summarize text using specified model"""
        if not self.models_trained:
            self.load_models()
        
        # Generate summary with length ratio
        if model_type == 'basic':
            summary = self.basic_model.predict(text, length_ratio=length_ratio)
        else:
            summary = self.attention_model.predict(text, length_ratio=length_ratio)
        
        # Post-process summary
        summary = summary.strip()
        if not summary:
            summary = "Unable to generate summary."
        
        return summary
    
    def get_text_stats(self, text):
        """Get statistics about the text"""
        words = len(text.split())
        chars = len(text)
        sentences = len(sent_tokenize(text))
        
        return {
            'word_count': words,
            'char_count': chars,
            'sentence_count': sentences
        }
    
    def calculate_compression_ratio(self, original_text, summary_text):
        """Calculate compression ratio"""
        original_words = len(original_text.split())
        summary_words = len(summary_text.split())
        
        if original_words == 0:
            return 0
        
        ratio = (original_words - summary_words) / original_words * 100
        return round(ratio, 2)
    
    def highlight_keywords(self, text, keywords):
        """Highlight keywords in text"""
        highlighted_text = text
        for keyword in keywords:
            highlighted_text = re.sub(
                f'\\b{re.escape(keyword)}\\b', 
                f'<mark>{keyword}</mark>', 
                highlighted_text, 
                flags=re.IGNORECASE
            )
        return highlighted_text