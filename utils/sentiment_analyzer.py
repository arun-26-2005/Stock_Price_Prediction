import yfinance as yf
import numpy as np

import os
import json

# Map user dataset names to Yahoo Finance symbols
TICKER_MAP = {
    'TCS': 'TCS.NS',
    'RELIANCE': 'RELIANCE.NS',
    'INFY': 'INFY.NS',
    'TATAMOTORS': 'TMCV.NS'
}

def load_custom_tickers():
    # Load custom tickers from data/custom_tickers.json
    registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "custom_tickers.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                custom = json.load(f)
                TICKER_MAP.update(custom)
        except Exception as e:
            print(f"Error loading custom tickers: {e}")

load_custom_tickers()

def fetch_recent_news(ticker_name):
    """
    Fetches recent news items for a stock using yfinance.
    Returns a list of dictionaries with headline metadata.
    """
    symbol = TICKER_MAP.get(ticker_name.upper(), ticker_name)
    print(f"Fetching recent news for {ticker_name} (symbol: {symbol}) via yfinance...")
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news
        if not news_items:
            print(f"No news articles found for {ticker_name}.")
            return []
            
        articles = []
        for item in news_items:
            # Handle nested 'content' structure in newer yfinance versions, fallback to flat
            content = item.get('content', item) if isinstance(item.get('content'), dict) else item
            
            title = content.get('title')
            
            # Parse publisher
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                publisher = provider.get('displayName', 'Unknown')
            else:
                publisher = content.get('publisher', 'Unknown')
                
            # Parse URL link
            canonical_url = content.get('canonicalUrl', {})
            if isinstance(canonical_url, dict):
                link = canonical_url.get('url', '')
            else:
                link = content.get('link', '')
                
            # Parse publish time
            publish_time = content.get('pubDate', content.get('providerPublishTime', 0))
            
            if title:
                articles.append({
                    'title': title,
                    'publisher': publisher,
                    'link': link,
                    'publish_time': publish_time
                })
        return articles
    except Exception as e:
        print(f"Error fetching news for {ticker_name}: {e}")
        return []

def analyze_sentiment(articles):
    """
    Analyzes the sentiment of a list of news articles.
    Attempts to use FinBERT first. If FinBERT fails to download or run,
    falls back to NLTK VADER.
    
    Returns:
        avg_score: consolidated sentiment score between -1.0 and +1.0
        detailed_sentiments: list of dictionaries detailing individual article sentiments
    """
    if not articles:
        return 0.0, []
        
    headlines = [a['title'] for a in articles]
    
    # Option 1: FinBERT (Financial BERT)
    try:
        print("Attempting to load FinBERT model...")
        from transformers import pipeline
        # Use ProsusAI/finbert (very stable, lightweight and optimized for finance)
        classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        print("FinBERT model loaded successfully.")
        
        results = classifier(headlines)
        
        scores = []
        detailed_sentiments = []
        
        for article, res in zip(articles, results):
            label = res['label'].lower() # 'positive', 'negative', 'neutral'
            score = res['score']
            
            # Map to [-1, 1] range
            if label == 'positive':
                num_score = score
            elif label == 'negative':
                num_score = -score
            else: # neutral
                num_score = 0.0
                
            scores.append(num_score)
            detailed_sentiments.append({
                'title': article['title'],
                'publisher': article['publisher'],
                'sentiment': label,
                'confidence': score,
                'score': num_score
            })
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, detailed_sentiments
        
    except Exception as e:
        print(f"FinBERT failed to initialize: {e}. Falling back to VADER...")
        
        # Option 2: NLTK VADER Fallback
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            nltk.download('vader_lexicon', quiet=True)
            
            analyzer = SentimentIntensityAnalyzer()
            scores = []
            detailed_sentiments = []
            
            for article in articles:
                vader_scores = analyzer.polarity_scores(article['title'])
                compound = vader_scores['compound'] # compound is between -1.0 and +1.0
                
                # Determine label
                if compound >= 0.05:
                    label = 'positive'
                elif compound <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                    
                scores.append(compound)
                detailed_sentiments.append({
                    'title': article['title'],
                    'publisher': article['publisher'],
                    'sentiment': label,
                    'confidence': abs(compound),
                    'score': compound
                })
                
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return avg_score, detailed_sentiments
            
        except Exception as ex:
            print(f"Fallback to VADER also failed: {ex}")
            return 0.0, []
