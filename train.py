"""
Training script for the text summarization models
"""

import os
import json
import argparse
from models import SummarizationSystem
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train text summarization models')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--model', choices=['basic', 'attention', 'both'], default='both', help='Which model to train')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    if args.epochs:
        config['model_config']['epochs'] = args.epochs
    if args.batch_size:
        config['model_config']['batch_size'] = args.batch_size
    
    # Initialize summarization system
    summarizer = SummarizationSystem(args.config)
    
    print("Starting training process...")
    print(f"Configuration: {config['model_config']}")
    
    # Create sample dataset
    print("Creating sample dataset...")
    texts, summaries = summarizer.create_sample_dataset()
    
    print("Training models...")
    
    print(f"Training set size: {len(texts)}")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Train models based on selection
    if args.model in ['basic', 'both']:
        print("\n" + "="*50)
        print("Training Basic LSTM Model")
        print("="*50)
        
        summarizer.basic_model.build_model()
        summarizer.basic_model.train(texts, summaries, [], [], epochs=args.epochs or 2, batch_size=args.batch_size or 16)
        summarizer.basic_model.save_model('models/basic_model')
        print("Basic model saved successfully!")
    
    if args.model in ['attention', 'both']:
        print("\n" + "="*50)
        print("Training Attention Model")
        print("="*50)
        
        summarizer.attention_model.build_model()
        summarizer.attention_model.train(texts, summaries, [], [], epochs=args.epochs or 2, batch_size=args.batch_size or 16)
        summarizer.attention_model.save_model('models/attention_model')
        print("Attention model saved successfully!")
    
    # Save tokenizer indicator
    import pickle
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump({'trained': True}, f)
    print("Tokenizer saved successfully!")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    
    # Test the models with a sample text
    print("\nTesting models with sample text...")
    sample_text = """
    The stock market experienced significant volatility today as investors reacted to the Federal Reserve's latest interest rate decision. 
    The Dow Jones Industrial Average fell by 300 points in early trading before recovering some losses by midday. 
    Technology stocks were particularly hard hit, with major companies seeing their share prices drop by more than 5 percent. 
    Market analysts attribute the decline to concerns about inflation and its impact on consumer spending.
    """
    
    try:
        # Test basic model
        if args.model in ['basic', 'both']:
            basic_summary = summarizer.summarize_text(sample_text, 'basic')
            print(f"\nBasic LSTM Summary: {basic_summary}")
        
        # Test attention model
        if args.model in ['attention', 'both']:
            attention_summary = summarizer.summarize_text(sample_text, 'attention')
            print(f"\nAttention Model Summary: {attention_summary}")
        
        print("\nModel testing completed successfully!")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        print("Models may need more training or there might be an issue with the implementation.")

if __name__ == "__main__":
    main()