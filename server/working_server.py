from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import datetime
import pymongo
from dateutil.parser import parse
from datetime import timedelta
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import requests

# Import your Gemini API wrapper
import google.generativeai as genai

# --- MongoDB Setup & Global Variables ---
MONGO_URI = "mongodb://localhost:27017"
client = pymongo.MongoClient(MONGO_URI)
db = client["crypto_data"]

# --- Global Variables ---
USERNAME = "default"
TOKEN = "BTC"
COINGECKECOINID = "bitcoin"
NEWS_COLLECTION = f"{USERNAME}_{TOKEN}"
SUGGESTIONS_COLLECTION = f"suggestions_{USERNAME}"

# --- Gemini API Function ---
def refine_markdown_with_gemini(prompt: str) -> str:
    API_KEY = 'AIzaSyDWki8PZMr6rotpuelz6xdXhw0h63YBYeQ'
    genai.configure(api_key=API_KEY)
    
    refined_prompt = (
        "Refine the following markdown text by adding engaging highlights, headers, bullet points, "
        "and additional stylistic elements. Ensure that the markdown is clear and well-structured:\n\n"
        f"{prompt}\n\n"
    )
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(refined_prompt)
    return response.text

# --- Data Loading Helpers ---
def load_market_data():
    cursor = db[MARKET_COLLECTION].find({"token": COINGECKECOINID}).sort("timestamp", pymongo.ASCENDING)
    docs = list(cursor)
    for doc in docs:
        dt = parse(doc["timestamp"])
        doc["timestamp"] = dt.replace(tzinfo=None)
    return docs

def load_news_data():
    cursor = db[NEWS_COLLECTION].find().sort("timestamp", pymongo.ASCENDING)
    docs = list(cursor)
    for doc in docs:
        dt = parse(doc["timestamp"])
        doc["timestamp"] = dt.replace(tzinfo=None)
    return docs

def load_whale_data():
    cursor = db[WHALE_COLLECTION].find().sort("timestamp", pymongo.ASCENDING)
    docs = list(cursor)
    for doc in docs:
        dt = parse(doc["timestamp"])
        doc["timestamp"] = dt.replace(tzinfo=None)
    return docs

def get_most_recent(doc_list, current_time):
    recent = None
    for doc in doc_list:
        if doc["timestamp"] <= current_time:
            recent = doc
        else:
            break
    return recent

def get_recent_data(load_func, days=3):
    docs = load_func()
    cutoff = datetime.datetime.utcnow() - timedelta(days=days)
    recent = [doc for doc in docs if doc["timestamp"] > cutoff]
    return recent

# --- Build State from Recent Data ---
def build_state_from_recent_data(recent_market, recent_news, recent_whale):
    if not recent_market:
        return None
    market_doc = recent_market[-1]
    market_time = market_doc["timestamp"]
    price_changes = market_doc.get("price_changes", [67000.0]*5)
    avg_price = np.mean(price_changes)
    price_volatility = np.std(price_changes)
    prices = [np.mean(doc.get("price_changes", [67000.0]*5)) for doc in recent_market]
    moving_avg = np.mean(prices)
    
    recent_news_doc = get_most_recent(recent_news, market_time)
    if recent_news_doc:
        sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
        base_sentiment = sentiment_map.get(recent_news_doc.get("sentiment", "neutral"), 0)
        intensity = recent_news_doc.get("sentiment_intensity", 0)
        news_signal = base_sentiment * intensity
    else:
        news_signal = 0.0
    
    recent_whale_doc = get_most_recent(recent_whale, market_time)
    if recent_whale_doc:
        whale_amount = recent_whale_doc.get("amount", 0)
        whale_signal = whale_amount / 1000.0
    else:
        whale_signal = 0.0
    
    avg_price_norm = avg_price / 100000.0
    volatility_norm = price_volatility / 100000.0
    moving_avg_norm = moving_avg / 100000.0
    tokens = 0.0  # assume tokens = 0 for prediction
    
    state = np.array([
        avg_price_norm,
        volatility_norm,
        news_signal,
        whale_signal,
        tokens,
        moving_avg_norm
    ], dtype=np.float32)
    return state

# --- Market Data Update ---
def get_market_data(token_id, days=90, currency="usd", retry_attempts=3):
    BASE_URL_COINGECKO = "https://api.coingecko.com/api/v3"
    API_KEY_COINGECKO = "CG-tSeVmpHWqyBqiYCHSQ4yN9se"
    url = f"{BASE_URL_COINGECKO}/coins/{token_id}/market_chart?vs_currency={currency}&days={days}"
    headers = {"x-cg-api-key": API_KEY_COINGECKO}
    for attempt in range(retry_attempts):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Successfully fetched {days} days of data for {token_id}")
            return data
        elif response.status_code == 429:
            print(f"⚠️ Rate limit exceeded for {token_id}! Waiting before retrying...")
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
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    db[MARKET_COLLECTION].insert_one(market_document)
    print(f"✅ Stored market data for {token} in MongoDB.")

def update_missing_market_data():
    docs = load_market_data()
    if not docs:
        print("No market data found.")
        return
    latest_timestamp = docs[-1]["timestamp"]
    now = datetime.datetime.utcnow()
    if now <= latest_timestamp:
        print("Market data is up-to-date.")
        return
    diff_days = (now - latest_timestamp).total_seconds() / (3600*24)
    new_data = get_market_data(COINGECKECOINID, days=diff_days, currency="usd")
    if new_data:
        store_market_data(COINGECKECOINID, new_data)
        print("Updated market data.")

# --- RL Agent Definitions (Dueling Double DQN with Noisy Networks) ---
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
        self.value_fc = NoisyLinear(128, 64)
        self.value = NoisyLinear(64, 1)
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
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        
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

def load_trained_agent(state_dim=6, action_dim=3):
    agent = DQNAgent(state_dim, action_dim)
    agent.model.load_state_dict(torch.load("trained_agent_chandru_BTC.pth", map_location=torch.device("cpu")))
    agent.model.eval()
    return agent

# --- Main Function to Retrieve Missing Data, Predict Next Action, and Generate Markdown ---
def get_trading_suggestion():
    update_missing_market_data()
    recent_market = get_recent_data(load_market_data, days=3)
    recent_news = get_recent_data(load_news_data, days=3)
    recent_whale = get_recent_data(load_whale_data, days=3)
    state = build_state_from_recent_data(recent_market, recent_news, recent_whale)
    if state is None:
        return "No recent market data available for prediction."
    agent = load_trained_agent()
    predicted_action = agent.select_action(state)
    action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
    action_text = action_map.get(predicted_action, "Unknown")
    news_headlines = "\n".join([f"- {doc.get('message','')}" for doc in recent_news[-3:]]) if recent_news else "No recent news."
    most_recent_market = recent_market[-1]
    avg_price = np.mean(most_recent_market.get("price_changes", [67000.0]))
    price_vol = np.std(most_recent_market.get("price_changes", [67000.0]))
    market_summary = f"Latest market average price: {avg_price:.2f}, volatility: {price_vol:.2f}"
    if recent_whale:
        most_recent_whale = recent_whale[-1]
        whale_summary = f"Recent whale transaction amount: {most_recent_whale.get('amount', 0)}"
    else:
        whale_summary = "No recent whale transactions."
    
    prompt = (
        f"**Recommended Action:** {action_text}\n\n"
        f"**Market Summary:** {market_summary}\n\n"
        f"**Recent News:**\n{news_headlines}\n\n"
        f"**Whale Activity:** {whale_summary}\n\n"
        f"Based on the last 3 days of data, the model suggests to **{action_text}**. "
        "Please provide a detailed markdown explanation justifying this recommendation based on the above data."
    )
    
    markdown_result = refine_markdown_with_gemini(prompt)
    # Save the suggestion with timestamp into a suggestions collection
    suggestion_doc = {
        "username": USERNAME,
        "token": TOKEN,
        "coingeckecoinid": COINGECKECOINID,
        "markdown": markdown_result,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    db[f"suggestions_{USERNAME}"].insert_one(suggestion_doc)
    return markdown_result

# --- FastAPI Setup ---
from fastapi import FastAPI, HTTPException
app = FastAPI()

class SuggestionResponse(BaseModel):
    suggestion: str

@app.get("/trading_suggestion", response_model=SuggestionResponse)
def trading_suggestion(username: str = Query(...), token: str = Query(...), coingeckecoinid: str = Query(...)):
    global USERNAME, TOKEN, COINGECKECOINID, NEWS_COLLECTION, WHALE_COLLECTION, MARKET_COLLECTION
    USERNAME = username
    TOKEN = token
    COINGECKECOINID = coingeckecoinid
    NEWS_COLLECTION = f"{USERNAME}_{TOKEN}"
    WHALE_COLLECTION = f"whale_{USERNAME}_{TOKEN}"
    MARKET_COLLECTION = f"market_{USERNAME}"
    try:
        suggestion_markdown = get_trading_suggestion()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return SuggestionResponse(suggestion=suggestion_markdown)

# --- Retrieve Markdown Suggestions ---
@app.get("/retrieve_markdown")
def retrieve_markdown(username: str = Query(...), token: str = Query(...)):
    collection_name = f"suggestions_{username}"
    latest_suggestion = db[collection_name].find_one(
        {"token": token}, sort=[("timestamp", pymongo.DESCENDING)]
    )
    
    if not latest_suggestion:
        raise HTTPException(status_code=404, detail="No markdown suggestions found for this user and token.")
    
    return {"markdown": latest_suggestion["markdown"]}

# To run the app, use:
# uvicorn <filename_without_py>:app --reload
