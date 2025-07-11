"""
Visualization script for the text summarization project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from models import SummarizationSystem
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_word_cloud(text, title="Word Cloud", save_path=None):
    """Create and display a word cloud"""
    
    # Remove common stop words
    stop_words = set(stopwords.words('english'))
    
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    text = ' '.join(words)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_text_statistics(original_text, summaries, model_names):
    """Plot text statistics comparison"""
    
    # Calculate statistics
    def get_stats(text):
        words = len(text.split())
        chars = len(text)
        sentences = len(re.split(r'[.!?]+', text))
        return words, chars, sentences
    
    original_stats = get_stats(original_text)
    summary_stats = [get_stats(summary) for summary in summaries]
    
    # Create comparison dataframe
    data = []
    data.append(['Original', original_stats[0], original_stats[1], original_stats[2]])
    
    for i, (model_name, stats) in enumerate(zip(model_names, summary_stats)):
        data.append([model_name, stats[0], stats[1], stats[2]])
    
    df = pd.DataFrame(data, columns=['Model', 'Words', 'Characters', 'Sentences'])
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot words
    bars1 = axes[0].bar(df['Model'], df['Words'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Word Count Comparison', fontweight='bold')
    axes[0].set_ylabel('Number of Words')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
    
    # Plot characters
    bars2 = axes[1].bar(df['Model'], df['Characters'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Character Count Comparison', fontweight='bold')
    axes[1].set_ylabel('Number of Characters')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom')
    
    # Plot sentences
    bars3 = axes[2].bar(df['Model'], df['Sentences'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_title('Sentence Count Comparison', fontweight='bold')
    axes[2].set_ylabel('Number of Sentences')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('text_statistics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_compression_ratio(original_text, summaries, model_names):
    """Plot compression ratio comparison"""
    
    original_words = len(original_text.split())
    
    compression_ratios = []
    for summary in summaries:
        summary_words = len(summary.split())
        ratio = ((original_words - summary_words) / original_words) * 100
        compression_ratios.append(ratio)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, compression_ratios, color=['#ff7f0e', '#2ca02c'])
    plt.title('Compression Ratio Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Compression Ratio (%)')
    plt.xlabel('Model')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('compression_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_keyword_frequency(texts, titles):
    """Plot keyword frequency comparison"""
    
    # Extract keywords from each text
    stop_words = set(stopwords.words('english'))
    
    all_keywords = []
    for text, title in zip(texts, titles):
        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        
        # Get top keywords
        word_freq = Counter(words)
        top_keywords = dict(word_freq.most_common(10))
        
        all_keywords.append((title, top_keywords))
    
    # Create subplots
    fig, axes = plt.subplots(1, len(all_keywords), figsize=(5*len(all_keywords), 6))
    
    if len(all_keywords) == 1:
        axes = [axes]
    
    for i, (title, keywords) in enumerate(all_keywords):
        words = list(keywords.keys())
        counts = list(keywords.values())
        
        axes[i].barh(words, counts)
        axes[i].set_title(f'{title} - Top Keywords', fontweight='bold')
        axes[i].set_xlabel('Frequency')
        
        # Add value labels
        for j, (word, count) in enumerate(zip(words, counts)):
            axes[i].text(count + 0.1, j, str(count), va='center')
    
    plt.tight_layout()
    plt.savefig('keyword_frequency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_similarity_matrix(texts, labels):
    """Plot similarity matrix between texts"""
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=True, 
                cmap='Blues', 
                vmin=0, 
                vmax=1,
                square=True)
    
    plt.title('Text Similarity Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_performance_dashboard():
    """Create a comprehensive dashboard of model performance"""
    
    # Initialize system
    summarizer = SummarizationSystem()
    
    # Check if models exist
    if not os.path.exists('models/tokenizer.pkl'):
        print("Models not found. Please train the models first.")
        return
    
    # Load models
    print("Loading models...")
    summarizer.load_models()
    
    # Test cases
    test_cases = [
        {
            "title": "Technology News",
            "text": """
            Apple Inc. announced its latest quarterly earnings today, reporting record revenue of $123.9 billion, 
            up 11% from the same period last year. The company's iPhone sales remained strong despite global 
            economic uncertainty, with the iPhone 15 series performing particularly well in international markets. 
            CEO Tim Cook highlighted the company's continued investment in artificial intelligence and machine learning 
            technologies, stating that AI will be a key driver of future growth. The company also announced plans 
            to expand its manufacturing capabilities in India and Vietnam to reduce dependence on Chinese production.
            """
        },
        {
            "title": "Climate Change Report",
            "text": """
            A new study published in Nature Climate Change reveals that global temperatures have risen by 1.2 degrees 
            Celsius since pre-industrial times, marking the fastest rate of warming in recorded history. The research, 
            conducted by an international team of climate scientists, analyzed temperature data from over 10,000 
            weather stations worldwide spanning the past 150 years. Scientists warn that without immediate action 
            to reduce greenhouse gas emissions, global temperatures could rise by 3-4 degrees Celsius by the end 
            of the century, leading to catastrophic environmental consequences.
            """
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nProcessing test case {i+1}: {test_case['title']}")
        
        original_text = test_case['text'].strip()
        
        # Generate summaries
        basic_summary = summarizer.summarize_text(original_text, 'basic')
        attention_summary = summarizer.summarize_text(original_text, 'attention')
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Word clouds
        create_word_cloud(original_text, f"Original Text - {test_case['title']}", 
                         f"wordcloud_original_{i+1}.png")
        create_word_cloud(basic_summary, f"Basic LSTM Summary - {test_case['title']}", 
                         f"wordcloud_basic_{i+1}.png")
        create_word_cloud(attention_summary, f"Attention Model Summary - {test_case['title']}", 
                         f"wordcloud_attention_{i+1}.png")
        
        # Statistics comparison
        plot_text_statistics(original_text, [basic_summary, attention_summary], 
                           ['Basic LSTM', 'Attention Model'])
        
        # Compression ratio
        plot_compression_ratio(original_text, [basic_summary, attention_summary], 
                             ['Basic LSTM', 'Attention Model'])
        
        # Keyword frequency
        plot_keyword_frequency([original_text, basic_summary, attention_summary], 
                              ['Original', 'Basic LSTM', 'Attention Model'])
        
        # Similarity matrix
        plot_similarity_matrix([original_text, basic_summary, attention_summary], 
                              ['Original', 'Basic LSTM', 'Attention Model'])

def plot_model_architecture():
    """Visualize model architectures"""
    
    # Initialize system
    summarizer = SummarizationSystem()
    
    # Build models
    basic_model = summarizer.basic_model.build_model()
    attention_model = summarizer.attention_model.build_model()
    
    # Plot model architectures
    try:
        from tensorflow.keras.utils import plot_model
        
        plot_model(basic_model, to_file='basic_model_architecture.png', 
                  show_shapes=True, show_layer_names=True, dpi=300)
        print("Basic model architecture saved as 'basic_model_architecture.png'")
        
        plot_model(attention_model, to_file='attention_model_architecture.png', 
                  show_shapes=True, show_layer_names=True, dpi=300)
        print("Attention model architecture saved as 'attention_model_architecture.png'")
        
    except ImportError:
        print("Cannot create model architecture plots. Install graphviz and pydot for this feature.")

def analyze_training_data():
    """Analyze the training data distribution"""
    
    # Initialize system
    summarizer = SummarizationSystem()
    
    # Get sample dataset
    texts, summaries = summarizer.create_sample_dataset()
    
    # Analyze text lengths
    text_lengths = [len(text.split()) for text in texts]
    summary_lengths = [len(summary.split()) for summary in summaries]
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Text length distribution
    axes[0, 0].hist(text_lengths, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Original Text Length Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Words')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(text_lengths), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(text_lengths):.1f}')
    axes[0, 0].legend()
    
    # Summary length distribution
    axes[0, 1].hist(summary_lengths, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Summary Length Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(summary_lengths), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(summary_lengths):.1f}')
    axes[0, 1].legend()
    
    # Length relationship
    axes[1, 0].scatter(text_lengths, summary_lengths, alpha=0.6)
    axes[1, 0].set_title('Text vs Summary Length Relationship', fontweight='bold')
    axes[1, 0].set_xlabel('Original Text Length (words)')
    axes[1, 0].set_ylabel('Summary Length (words)')
    
    # Compression ratio distribution
    compression_ratios = [((len(text.split()) - len(summary.split())) / len(text.split())) * 100 
                         for text, summary in zip(texts, summaries)]
    
    axes[1, 1].hist(compression_ratios, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Compression Ratio Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Compression Ratio (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(compression_ratios), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(compression_ratios):.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training data statistics:")
    print(f"Number of samples: {len(texts)}")
    print(f"Average text length: {np.mean(text_lengths):.1f} words")
    print(f"Average summary length: {np.mean(summary_lengths):.1f} words")
    print(f"Average compression ratio: {np.mean(compression_ratios):.1f}%")

def main():
    """Main visualization function"""
    
    print("Text Summarization Visualization Dashboard")
    print("="*50)
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    os.chdir('visualizations')
    
    try:
        # Analyze training data
        print("\n1. Analyzing training data...")
        analyze_training_data()
        
        # Plot model architectures
        print("\n2. Creating model architecture plots...")
        plot_model_architecture()
        
        # Create performance dashboard
        print("\n3. Creating model performance dashboard...")
        create_model_performance_dashboard()
        
        print("\n4. All visualizations have been saved to the 'visualizations' directory!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure all dependencies are installed and models are trained.")

if __name__ == "__main__":
    main()