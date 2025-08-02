#!/usr/bin/env python3
from transformers import pipeline
import torch

# Create sentiment analyzer with PyTorch backend
print("Loading FinBERT model with PyTorch...")
sentiment = pipeline('sentiment-analysis', 
                    model='ProsusAI/finbert',
                    framework='pt')  # Force PyTorch to avoid TF issues

# Test on sample news
news = [
    'Apple reports record quarterly earnings', 
    'Market volatility continues amid recession fears',
    'Tesla stock surges on strong delivery numbers'
]

print("Testing sentiment analysis:")
results = sentiment(news)
for text, result in zip(news, results):
    print(f"  '{text}' -> {result['label']} ({result['score']:.3f})")

print("✅ Sentiment analysis working!")