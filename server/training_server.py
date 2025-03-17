import requests
import tweepy
import pymongo
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from dateutil.parser import parse
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import os

# ==========================================
# CONFIGURATION & MONGODB SETUP
# ==========================================
# API Keys (Replace with your actual keys)
CRYPTOPANIC_API_KEY = "14bb5475e7a1ce9af2acaaf94698ed978b841d65"
TWITTER_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAPzRzwEAAAAA5eizIBQJ%2Fkv4ufxfJsAqyEL0fbQ%3D01ZBa0zGAy8RKTFoUWeJx6RguVRZ7Svv7rqqQuQ0FYusurVq2g"
BITQUERY_API_KEY = "ory_at_qHMqpHwt4mdJM5cupri1oc7zdGojjsIc_Pm8f3Xvfco.YMWzcTJ3RvtRxMgOEpSSwyKxl7sN3iCsHIcEXJxi6c0"
API_KEY_COINGECKO = "CG-tSeVmpHWqyBqiYCHSQ4yN9se"  # CoinGecko API key if needed

# Endpoints & URIs
BITQUERY_ENDPOINT = "https://graphql.bitquery.io"
BASE_URL_COINGECKO = "https://api.coingecko.com/api/v3"
MONGO_URI = "mongodb://localhost:27017"

# MongoDB Connection & Database
client = pymongo.MongoClient(MONGO_URI)
db = client["crypto_data"]

# User & Coin Configuration
USERNAME = "chandru"    # Replace with your username
TOKEN = "BTC"                 # Symbol for news/tweets & whale data
COINGECKO_COIN_ID = "bitcoin" # CoinGecko uses 'bitcoin' for BTC

# Collections names
NEWS_COLLECTION = f"{USERNAME}_{TOKEN}"           # For news & tweets
WHALE_COLLECTION = f"whale_{USERNAME}_{TOKEN}"      # For whale transactions
MARKET_COLLECTION = f"market_{USERNAME}"            # For market data

# ==========================================
# LOAD CRYPTOBERT MODEL FOR SENTIMENT CLASSIFICATION
# ==========================================
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding="max_length")


# ==========================================
# FETCH & STORE CRYPTO NEWS (CryptoPanic) & TWEETS (Twitter)
# ==========================================
def fetch_crypto_news(token):
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&currencies={token}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    print("Error fetching CryptoPanic news:", response.json())
    return []

def fetch_tweets(token, count=10):
    """Fetches recent tweets related to the given token using Twitter API with rate limit handling."""
    count = max(count, 10)  # Ensure count is at least 10 (Twitter API requirement)
    client_twitter = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    query = f"#{token} OR {token} crypto -is:retweet -is:reply lang:en"
    try:
        tweets = client_twitter.search_recent_tweets(
            query=query,
            max_results=count,
            tweet_fields=["created_at", "text"]
        )
        return tweets.data if tweets and tweets.data else []
    except tweepy.TooManyRequests as e:
        print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
        time.sleep(60)
        return fetch_tweets(token, count)
    except Exception as e:
        print("An error occurred while fetching tweets:", e)
        return []

def classify_and_store_news(news_data):
    structured_news = []
    for news in news_data:
        message = news.get("title", "")
        url = news.get("url", "")
        timestamp = news.get("published_at", datetime.utcnow().isoformat())
        # Classify sentiment using CryptoBERT
        sentiment_result = pipe(message)[0]
        sentiment = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        structured_entry = {
            "source": "CryptoPanic",
            "message": message,
            "sentiment": sentiment,
            "sentiment_intensity": sentiment_score,
            "relevance_score": news.get("relevance", 0.9),
            "timestamp": timestamp,
            "url": url
        }
        structured_news.append(structured_entry)
    if structured_news:
        db[NEWS_COLLECTION].insert_many(structured_news)
        print(f"Stored {len(structured_news)} news articles in MongoDB.")

def classify_and_store_tweets(tweets):
    structured_tweets = []
    for tweet in tweets:
        message = tweet.text
        timestamp = tweet.created_at.isoformat()
        sentiment_result = pipe(message)[0]
        sentiment = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        structured_entry = {
            "source": "Twitter",
            "message": message,
            "sentiment": sentiment,
            "sentiment_intensity": sentiment_score,
            "relevance_score": 0.9,  # fixed relevance for tweets
            "timestamp": timestamp
        }
        structured_tweets.append(structured_entry)
    if structured_tweets:
        db[NEWS_COLLECTION].insert_many(structured_tweets)
        print(f"Stored {len(structured_tweets)} tweets in MongoDB.")

# ==========================================
# FETCH & STORE WHALE TRANSACTIONS (Bitquery)
# ==========================================
def fetch_transactions(since, till):
    query = f"""
    {{
      ethereum(network: ethereum) {{
        transfers(
          options: {{desc: "block.timestamp.time", limit: 1000}} 
          amount: {{gt: 100000}} 
          date: {{since: "{since}", till: "{till}"}}
        ) {{
          transaction {{
            hash
          }}
          sender {{
            address
          }}
          receiver {{
            address
          }}
          currency {{
            symbol
          }}
          amount
          block {{
            timestamp {{
              time
            }}
          }}
        }}
      }}
    }}
    """
    headers = {
        "Authorization": f"Bearer {BITQUERY_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(BITQUERY_ENDPOINT, json={"query": query}, headers=headers)
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        print(f"Error: Unable to parse API response for {since} - {till}")
        return []
    if "errors" in response_data:
        print(f"API Error for {since} - {till}: {response_data['errors']}")
        return []
    return response_data.get("data", {}).get("ethereum", {}).get("transfers", [])

def process_and_store_whale_transactions():
    start_date = datetime.utcnow() - timedelta(days=90)
    end_date = datetime.utcnow()
    chunk_size = timedelta(days=7)
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        print(f"Fetching transactions from {current_start} to {current_end}...")
        transactions = fetch_transactions(
            current_start.strftime("%Y-%m-%dT00:00:00Z"),
            current_end.strftime("%Y-%m-%dT23:59:59Z")
        )
        documents = []
        for tx in transactions:
            wallet_address = tx.get("sender", {}).get("address", "Unknown")
            transaction_type = "buy"  # Default; adjust as needed
            token_symbol = tx.get("currency", {}).get("symbol", TOKEN).upper()
            try:
                amount = float(tx.get("amount", 0))
            except (ValueError, TypeError):
                amount = 0.0
            conversion = 67000 if token_symbol == "BTC" else 1.0
            usd_value = round(amount * conversion, 2)
            timestamp = tx.get("block", {}).get("timestamp", {}).get("time", None)
            if timestamp:
                try:
                    timestamp = parse(timestamp).isoformat()
                except Exception:
                    timestamp = datetime.utcnow().isoformat()
            else:
                timestamp = datetime.utcnow().isoformat()
            doc = {
                "wallet_address": wallet_address,
                "transaction_type": transaction_type,
                "token": token_symbol,
                "amount": amount,
                "usd_value": usd_value,
                "timestamp": timestamp
            }
            documents.append(doc)
        if documents:
            db[WHALE_COLLECTION].insert_many(documents)
            print(f"Stored {len(documents)} whale transactions for {current_start} - {current_end}")
        else:
            print(f"No transactions found for {current_start} - {current_end}")
        current_start += chunk_size
        time.sleep(10)  # Avoid rate limits
    print(f"Done! Whale transaction data stored in MongoDB collection '{WHALE_COLLECTION}'.")

# ==========================================
# FETCH & STORE MARKET DATA (CoinGecko)
# ==========================================
def get_market_data(token_id, days=90, currency="usd", retry_attempts=3):
    url = f"{BASE_URL_COINGECKO}/coins/{token_id}/market_chart?vs_currency={currency}&days={days}"
    headers = {"x-cg-api-key": API_KEY_COINGECKO}
    for attempt in range(retry_attempts):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Successfully fetched {days} days of data for {token_id}")
            return data
        elif response.status_code == 429:
            print(f"⚠️ Rate limit exceeded for {token_id}! Waiting before retrying ({attempt+1}/{retry_attempts})...")
            time.sleep(30)
        else:
            print(f"Error fetching data for {token_id}:", response.json())
            return None
    print(f"Failed to fetch data for {token_id} after {retry_attempts} attempts.")
    return None

def store_market_data(token, data):
    try:
        price_list = [round(entry[1], 2) for entry in data["prices"][:5]]
    except (KeyError, IndexError):
        print(f"Error processing price data for {token}")
        return
    market_document = {
        "token": token,
        "timeframe": "minute",
        "price_changes": price_list,
        "timestamp": datetime.utcnow().isoformat()
    }
    db[MARKET_COLLECTION].insert_one(market_document)
    print(f"✅ Stored market data for {token} in MongoDB.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- News & Tweets ---
    print("Fetching and processing news...")
    news_data = fetch_crypto_news(TOKEN)
    classify_and_store_news(news_data)

    print("Fetching and processing tweets...")
    tweets_data = fetch_tweets(TOKEN, count=5)  # Will use at least 10 due to API limits
    classify_and_store_tweets(tweets_data)

    # --- Whale Transactions ---
    print("Fetching and processing whale transactions...")
    process_and_store_whale_transactions()

    # --- Market Data ---
    # For market data, we use CoinGecko's coin ID (e.g., "bitcoin" for BTC)
    print("Fetching and processing market data...")
    market_data = get_market_data(COINGECKO_COIN_ID)
    if market_data:
        store_market_data(COINGECKO_COIN_ID, market_data)

    print("Done! All data stored in MongoDB.")

import random
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pymongo
from dateutil.parser import parse

# ==========================================
# MongoDB Setup & Configuration
# ==========================================
MONGO_URI = "mongodb://localhost:27017"
client = pymongo.MongoClient(MONGO_URI)
db = client["crypto_data"]

# User & Coin Configuration
USERNAME = "chandru"      # Replace with your username
TOKEN = "BTC"                   # Trading coin symbol
COINGECKO_COIN_ID = "bitcoin"   # For market data (CoinGecko coin id)
MARKET_COLLECTION = f"market_{USERNAME}"           # Market data documents
NEWS_COLLECTION = f"{USERNAME}_{TOKEN}"             # News & tweets documents
WHALE_COLLECTION = f"whale_{USERNAME}_{TOKEN}"        # Whale transactions

# ==========================================
# Helper Functions to Load Data from MongoDB
# ==========================================
def load_market_data():
    """Load market data sorted by timestamp for the coin."""
    cursor = db[MARKET_COLLECTION].find({"token": COINGECKO_COIN_ID}).sort("timestamp", pymongo.ASCENDING)
    docs = list(cursor)
    for doc in docs:
        dt = parse(doc["timestamp"])
        doc["timestamp"] = dt.replace(tzinfo=None)
    return docs

def load_news_data():
    """Load news/tweet data sorted by timestamp for the coin."""
    cursor = db[NEWS_COLLECTION].find().sort("timestamp", pymongo.ASCENDING)
    docs = list(cursor)
    for doc in docs:
        dt = parse(doc["timestamp"])
        doc["timestamp"] = dt.replace(tzinfo=None)
    return docs

def load_whale_data():
    """Load whale transaction data sorted by timestamp for the coin."""
    cursor = db[WHALE_COLLECTION].find().sort("timestamp", pymongo.ASCENDING)
    docs = list(cursor)
    for doc in docs:
        dt = parse(doc["timestamp"])
        doc["timestamp"] = dt.replace(tzinfo=None)
    return docs

def get_most_recent(doc_list, current_time):
    """
    Return the most recent document from a sorted list (by timestamp)
    with timestamp <= current_time; if none exists, return None.
    """
    recent = None
    for doc in doc_list:
        if doc["timestamp"] <= current_time:
            recent = doc
        else:
            break
    return recent

# ==========================================
# Real Data Trading Environment (Reworked)
# ==========================================
class RealDataTradingEnv:
    """
    Trading environment for a single coin (BTC) using actual data from MongoDB.
    
    The environment uses continuous market data for state transitions.
    Sparse event data (news and whale transactions) is handled by retaining the last
    observed event signal if no new event is available.
    
    State vector (6 features):
      0. Normalized average market price (avg price / 100,000)
      1. Normalized market volatility (std dev / 100,000)
      2. News sentiment signal (last observed or updated when new news arrives)
      3. Whale signal (last observed or updated when new whale event arrives)
      4. Current token holdings (raw value)
      5. Normalized moving average of recent prices
      
    Actions:
      0: Hold  
      1: Buy (spend 10% of cash)  
      2: Sell (sell 10% of token holdings)
      
    Reward:
      Change in portfolio value.
    """
    def __init__(self, max_steps=100, window_size=3):
        self.max_steps = max_steps
        self.initial_cash = 100000.0
        self.window_size = window_size
        
        # Load actual data from MongoDB
        self.market_docs = load_market_data()       # Continuous market data
        self.news_docs = load_news_data()             # Sparse news/tweet data
        self.whale_docs = load_whale_data()           # Sparse whale transactions
        
        # For persistent event signals
        self.last_news_signal = 0.0
        self.last_whale_signal = 0.0
        
        # Pointer for market time series
        self.current_market_index = 0
        self.reset()
    
    def reset(self):
        self.cash = self.initial_cash
        self.tokens = 0.0
        self.current_step = 0
        self.done = False
        self.prev_portfolio_value = self.cash
        self.portfolio_history = [self.cash]
        self.current_market_index = 0
        self.recent_prices = []
        # Reset persistent event signals
        self.last_news_signal = 0.0
        self.last_whale_signal = 0.0
        return self._get_state()
    
    def _get_state(self):
        if self.current_market_index >= len(self.market_docs):
            self.done = True
            return np.zeros(6, dtype=np.float32)
        
        market_doc = self.market_docs[self.current_market_index]
        market_time = market_doc["timestamp"]
        price_changes = market_doc.get("price_changes", [])
        if not price_changes:
            price_changes = [67000.0] * 5
        
        avg_price = np.mean(price_changes)
        price_volatility = np.std(price_changes)
        print(f"Market time: {market_time}, Avg price: {avg_price}, Volatility: {price_volatility}")
        
        self.recent_prices.append(avg_price)
        if len(self.recent_prices) > self.window_size:
            self.recent_prices.pop(0)
        moving_avg = np.mean(self.recent_prices)
        
        # News signal
        recent_news = get_most_recent(self.news_docs, market_time)
        if recent_news:
            sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
            base_sentiment = sentiment_map.get(recent_news.get("sentiment", "neutral"), 0)
            intensity = recent_news.get("sentiment_intensity", 0)
            self.last_news_signal = base_sentiment * intensity
        news_signal = self.last_news_signal
        print("News signal:", news_signal)
        
        # Whale signal
        recent_whale = get_most_recent(self.whale_docs, market_time)
        if recent_whale:
            whale_amount = recent_whale.get("amount", 0)
            self.last_whale_signal = whale_amount / 1000.0
        whale_signal = self.last_whale_signal
        print("Whale signal:", whale_signal)
        
        avg_price_norm = avg_price / 100000.0
        volatility_norm = price_volatility / 100000.0
        moving_avg_norm = moving_avg / 100000.0
        
        state = np.array([
            avg_price_norm,
            volatility_norm,
            news_signal,
            whale_signal,
            self.tokens,
            moving_avg_norm
        ], dtype=np.float32)
        return state

    
    def step(self, action):
        if self.done:
            # Instead of raising an exception, we simply return terminal state
            return np.zeros(6, dtype=np.float32), 0.0, True, {"portfolio_value": self.portfolio_history[-1]}
        
        market_doc = self.market_docs[self.current_market_index]
        price_changes = market_doc.get("price_changes", [])
        current_price = np.mean(price_changes)
        
        # Execute action: 1 = Buy, 2 = Sell, 0 = Hold
        if action == 1:  # Buy: spend 10% of cash
            spend = 0.10 * self.cash
            tokens_bought = spend / current_price
            self.tokens += tokens_bought
            self.cash -= spend
        elif action == 2:  # Sell: sell 10% of holdings
            tokens_to_sell = 0.10 * self.tokens
            self.tokens -= tokens_to_sell
            self.cash += tokens_to_sell * current_price
        
        portfolio_value = self.cash + self.tokens * current_price
        reward = portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = portfolio_value
        self.portfolio_history.append(portfolio_value)
        
        self.current_market_index += 1
        self.current_step += 1
        if self.current_market_index >= len(self.market_docs) or self.current_step >= self.max_steps:
            self.done = True
        
        next_state = self._get_state()
        next_state[4] = self.tokens  # Ensure token holdings are updated
        info = {"portfolio_value": portfolio_value, "cash": self.cash, "tokens": self.tokens}
        return next_state, reward, self.done, info
    
    def compute_risk_metrics(self):
        values = np.array(self.portfolio_history)
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks
        max_drawdown = np.max(drawdowns)
        returns = np.diff(values) / values[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 0 else 0
        return max_drawdown, sharpe

# ==========================================
# Experience Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ==========================================
# Dueling Double DQN Agent with Noisy Networks
# ==========================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            
        self.sigma_init = sigma_init
        self.reset_parameters()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init)
    
    def forward(self, x):
        weight_epsilon = torch.randn(self.out_features, self.in_features, device=x.device)
        bias_epsilon = torch.randn(self.out_features, device=x.device) if self.bias_mu is not None else None
        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        bias = self.bias_mu + self.bias_sigma * bias_epsilon if self.bias_mu is not None else None
        return nn.functional.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = NoisyLinear(state_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        # Value stream
        self.value_fc = NoisyLinear(128, 64)
        self.value = NoisyLinear(64, 1)
        # Advantage stream
        self.advantage_fc = NoisyLinear(128, 64)
        self.advantage = NoisyLinear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        
        self.current_step = 0
        self.epsilon = epsilon_start
        
        self.model = DuelingDQN(state_dim, action_dim)
        self.target_model = DuelingDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.update_target()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def select_action(self, state):
        self.current_step += 1
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                       np.exp(-1. * self.current_step / self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.max(1)[1].item()
    
    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state      = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action     = torch.LongTensor(action)
        reward     = torch.FloatTensor(reward)
        done       = torch.FloatTensor(done)
        
        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.model(next_state)
        next_actions = next_q_values.max(1)[1]
        next_target_q_values = self.target_model(next_state)
        next_q_value = next_target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = nn.functional.mse_loss(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ==========================================
# Training Loop
# ==========================================
def train_agent(num_episodes=100, max_steps=100):
    env = RealDataTradingEnv(max_steps=max_steps, window_size=3)
    state_dim = 6   # [avg_price_norm, volatility_norm, news_signal, whale_signal, tokens, moving_avg_norm]
    action_dim = 3  # [hold, buy, sell]
    
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []
    risk_metrics = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        # Initialize info in case environment finishes immediately
        info = {"portfolio_value": env.portfolio_history[-1]}
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step(batch_size=32)
            state = next_state
            total_reward += reward
            if done:
                break
        
        max_drawdown, sharpe = env.compute_risk_metrics()
        risk_metrics.append((max_drawdown, sharpe))
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            agent.update_target()
        
        portfolio_value = info.get('portfolio_value', env.portfolio_history[-1])
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f}, Portfolio: {portfolio_value:.2f}, Max DD: {max_drawdown:.2%}, Sharpe: {sharpe:.2f}")
    
    print("Training completed.")
    torch.save(agent.model.state_dict(), "trained_agent.pth")

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os

app = FastAPI()

class TrainResponse(BaseModel):
    message: str
    model_file: str

@app.get("/train_model", response_model=TrainResponse)
def train_model(
    username: str = Query(..., description="The username"),
    token: str = Query(..., description="The token symbol (e.g., BTC)"),
    coingeckecoinid: str = Query(..., description="The CoinGecko coin id (e.g., bitcoin)"),
    num_episodes: int = Query(100, description="Number of training episodes"),
    max_steps: int = Query(100, description="Max steps per episode")
):
    # Update global configuration variables
    global USERNAME, TOKEN, COINGECKECOINID, NEWS_COLLECTION, WHALE_COLLECTION, MARKET_COLLECTION
    USERNAME = username
    TOKEN = token
    COINGECKECOINID = coingeckecoinid
    NEWS_COLLECTION = f"{USERNAME}_{TOKEN}"
    WHALE_COLLECTION = f"whale_{USERNAME}_{TOKEN}"
    MARKET_COLLECTION = f"market_{USERNAME}"
    
    try:
        # Call the training function (assumes train_agent is defined above)
        train_agent(num_episodes=num_episodes, max_steps=max_steps)
        
        # Rename the model file to include the username and token so that the suggestion API can use it later
        new_model_filename = f"trained_agent_{USERNAME}_{TOKEN}.pth"
        os.rename("trained_agent.pth", new_model_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return TrainResponse(message="Training completed successfully.", model_file=new_model_filename)

# To run the app, use: uvicorn <this_filename_without_.py>:app --reload