from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly
import json
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, Flatten
import random
import os
import hashlib
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.optimize import minimize
import nltk
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import uuid
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import io
import base64
import finnhub
from email_validator import validate_email, EmailNotValidError
from flask_mail import Mail, Message
from dotenv import load_dotenv
load_dotenv()       # loads .env into os.environ

import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler  # <-- NEW import
import os

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.metrics import mean_absolute_error, r2_score
import random
import io
import base64
import yfinance as yf

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///cosmic_finance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail settings for password reset
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS') != 'False'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER') or 'noreply@cosmicfinance.ai'


# Create 'logs' directory if it doesn't exist
if not os.path.exists('logs'):
    os.mkdir('logs')

# General log handler: logs INFO level and above to app.log
general_handler = ConcurrentRotatingFileHandler(
    'logs/app.log', maxBytes=10240, backupCount=10
)
general_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
general_handler.setLevel(logging.INFO)
app.logger.addHandler(general_handler)

# Error log handler: logs ERROR level and above to error.log
error_handler = ConcurrentRotatingFileHandler(
    'logs/error.log', maxBytes=10240, backupCount=10
)
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
error_handler.setLevel(logging.ERROR)
app.logger.addHandler(error_handler)

# Add dedicated errors.log handler for comprehensive error tracking
errors_handler = ConcurrentRotatingFileHandler(
    'errors.log', maxBytes=10240, backupCount=10
)
errors_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
errors_handler.setLevel(logging.ERROR)
app.logger.addHandler(errors_handler)

app.logger.info('Error logging enabled to errors.log')

# Set the overall logger level to INFO
app.logger.setLevel(logging.INFO)
app.logger.info('Application startup')



# Initialize extensions
db = SQLAlchemy(app)
mail = Mail(app)

# Initialize APIs (use your own API keys in production)
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY') or 'demo_finnhub_api_key'
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY') or 'demo_alpha_vantage_key'

# Initialize Finnhub client with error handling
try:
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    # Test connection
    FINNHUB_AVAILABLE = True
except Exception as e:
    finnhub_client = None
    FINNHUB_AVAILABLE = False
    print(f"Error initializing Finnhub API: {str(e)}")

# Download NLTK data for sentiment analysis
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# -------------------------------
# Database Models for User Accounts
# -------------------------------

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    favorites = db.relationship('Favorite', backref='user', lazy='dynamic')
    portfolios = db.relationship('Portfolio', backref='user', lazy='dynamic')
    reset_token = db.Column(db.String(128))
    reset_token_expiry = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self):
        """Generate a password reset token valid for 24 hours"""
        self.reset_token = str(uuid.uuid4())
        self.reset_token_expiry = dt.datetime.utcnow() + dt.timedelta(hours=24)
        db.session.commit()
        return self.reset_token

    def verify_reset_token(self, token):
        """Verify that the reset token is valid"""
        if self.reset_token != token:
            return False
        if dt.datetime.utcnow() > self.reset_token_expiry:
            return False
        return True

# Favorite Tickers Model
class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Portfolio Model
class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    holdings = db.relationship('Holding', backref='portfolio', lazy='dynamic')

# Portfolio Holdings Model
class Holding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20))
    quantity = db.Column(db.Float)
    purchase_price = db.Column(db.Float)
    purchase_date = db.Column(db.DateTime)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'))
    notes = db.Column(db.Text)
    last_updated = db.Column(db.DateTime, default=dt.datetime.utcnow)

# User Tutorial Progress Model
class TutorialProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    tutorial_name = db.Column(db.String(100))
    completed = db.Column(db.Boolean, default=False)
    last_step = db.Column(db.Integer, default=0)
    completed_at = db.Column(db.DateTime)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------------------
# Portfolio Optimizer Class
# -------------------------------

class PortfolioOptimizer:
    def __init__(self):
        # Risk preference parameters
        self.risk_parameters = {
            'low': {
                'max_volatility': 0.15,  # 15% maximum annual volatility
                'min_sharpe_ratio': 0.5,
                'return_weight': 0.3,    # Lower weight on returns
                'risk_weight': 0.7       # Higher weight on risk minimization
            },
            'moderate': {
                'max_volatility': 0.25,  # 25% maximum annual volatility
                'min_sharpe_ratio': 0.7,
                'return_weight': 0.5,    # Equal weight on returns and risk
                'risk_weight': 0.5
            },
            'high': {
                'max_volatility': 0.40,  # 40% maximum annual volatility
                'min_sharpe_ratio': 0.8,
                'return_weight': 0.8,    # Higher weight on returns
                'risk_weight': 0.2       # Lower weight on risk minimization
            }
        }
        
    def optimize_portfolio(self, tickers, risk_preference='moderate'):
        """
        Optimize a portfolio of assets based on historical data and risk preference
        
        Parameters:
        - tickers: List of stock tickers
        - risk_preference: 'low', 'moderate', or 'high'
        
        Returns:
        - Portfolio optimization results including weights and performance metrics
        """
        try:
            import datetime as dt
            import numpy as np
            import pandas as pd
            import yfinance as yf
            from scipy.optimize import minimize
            from flask import current_app as app
            
            # Validate risk_preference
            if risk_preference.lower() not in self.risk_parameters:
                app.logger.warning(f"Invalid risk preference: {risk_preference}. Using 'moderate' instead.")
                risk_preference = 'moderate'
                
            risk_params = self.risk_parameters[risk_preference.lower()]
            
            # Get historical data for tickers
            end_date = dt.date.today()
            start_date = end_date - dt.timedelta(days=365 * 5)  # 5 years of data for better estimates
            
            app.logger.info(f"Optimizing portfolio for tickers: {tickers} with risk preference: {risk_preference}")
            
            # Download data
            data = yf.download(tickers, start=start_date, end=end_date)
            
            # Check if we have data
            if data.empty:
                app.logger.error("No data found for the given tickers")
                return None
                
            # Determine which price column to use - handle different data structures
            price_data = None
            
            # Check if we have multi-level columns (multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                # For multiple tickers, we'll have columns like ('Adj Close', 'AAPL')
                if 'Adj Close' in data.columns.levels[0]:
                    price_data = data['Adj Close']
                elif 'Close' in data.columns.levels[0]:
                    price_data = data['Close']
                else:
                    error_msg = "Could not find price data columns in multi-ticker data"
                    app.logger.error(error_msg)
                    raise Exception(error_msg)
            else:
                # For single ticker, columns are simple strings
                if 'Adj Close' in data.columns:
                    price_data = data[['Adj Close']]
                elif 'Close' in data.columns:
                    price_data = data[['Close']]
                else:
                    error_msg = "Could not find price data columns in single-ticker data"
                    app.logger.error(error_msg)
                    raise Exception(error_msg)
                    
            # If only one ticker, ensure data is in the right format
            if len(tickers) == 1 and not isinstance(price_data, pd.DataFrame):
                price_data = pd.DataFrame(price_data, columns=tickers)
                
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            app.logger.info(f"Portfolio contains {len(tickers)} assets")
            
            # Set risk-free rate (typically treasury yield)
            risk_free_rate = 0.02/252  # Assuming 2% annual risk-free rate, converted to daily
            
            # Define the objective function based on risk preference
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized return
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
                sharpe_ratio = (portfolio_return - risk_free_rate * 252) / portfolio_volatility
                
                # Risk preference specific components
                return_component = -portfolio_return  # Negative because we're minimizing
                risk_component = portfolio_volatility
                
                # Apply risk preferences weights to return and risk components
                weighted_objective = (risk_params['return_weight'] * return_component + 
                                     risk_params['risk_weight'] * risk_component)
                
                # Add constraints as penalties for violating risk preferences
                volatility_penalty = max(0, portfolio_volatility - risk_params['max_volatility']) * 100
                sharpe_penalty = max(0, risk_params['min_sharpe_ratio'] - sharpe_ratio) * 10
                
                return weighted_objective + volatility_penalty + sharpe_penalty
            
            # Constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            # Bounds: each weight between 0 and 1
            bounds = tuple((0, 1) for _ in range(len(tickers)))
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0/len(tickers)] * len(tickers))
            
            # Perform optimization
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if not result['success']:
                app.logger.warning(f"Optimization failed: {result.get('message', 'Unknown error')}. Using equally weighted portfolio.")
                weights = initial_weights
            else:
                weights = result['x']
                app.logger.info(f"Optimization successful")
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - risk_free_rate * 252) / portfolio_volatility
            
            # Calculate individual asset contributions
            portfolio_data = []
            for i, ticker in enumerate(tickers):
                # Get mean return and volatility for this asset
                ticker_return = mean_returns.iloc[i] if isinstance(mean_returns, pd.Series) else mean_returns[i]
                ticker_volatility = np.sqrt(cov_matrix.iloc[i, i] if isinstance(cov_matrix, pd.DataFrame) else cov_matrix[i, i])
                
                # Calculate percentage contribution to risk
                if portfolio_volatility > 0:
                    marginal_contribution = weights[i] * (cov_matrix.iloc[i] if isinstance(cov_matrix, pd.DataFrame) else cov_matrix[i])
                    risk_contribution = np.dot(marginal_contribution, weights) / portfolio_volatility
                    risk_contribution_pct = risk_contribution / portfolio_volatility * 100
                else:
                    risk_contribution_pct = 0
                
                portfolio_data.append({
                    'ticker': ticker,
                    'weight': weights[i] * 100,  # Convert to percentage
                    'expected_return': ticker_return * 252 * 100,  # Annualized percentage
                    'volatility': ticker_volatility * np.sqrt(252) * 100,  # Annualized percentage
                    'risk_contribution': risk_contribution_pct
                })
                
            # Sort by weight, descending
            portfolio_data.sort(key=lambda x: x['weight'], reverse=True)
            
            # Calculate risk profile metrics
            risk_profile = {
                'targeted_max_volatility': risk_params['max_volatility'] * 100,  # As percentage
                'targeted_min_sharpe': risk_params['min_sharpe_ratio'],
                'return_risk_preference': f"{risk_params['return_weight']}/{risk_params['risk_weight']}"
            }
            
            # Log the results
            app.logger.info(f"Portfolio return: {portfolio_return * 100:.2f}%, "
                           f"Portfolio volatility: {portfolio_volatility * 100:.2f}%, "
                           f"Sharpe ratio: {sharpe_ratio:.2f}")
            
            # Return results
            return {
                'portfolio_data': portfolio_data,
                'portfolio_return': portfolio_return * 100,  # Convert to percentage
                'portfolio_volatility': portfolio_volatility * 100,  # Convert to percentage
                'sharpe_ratio': sharpe_ratio,
                'risk_preference': risk_preference,
                'risk_profile': risk_profile
            }
            
        except Exception as e:
            error_msg = f"Portfolio optimization error: {str(e)}"
            app.logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

# -------------------------------
# Helper Functions for Stock Prediction
# -------------------------------

def create_sequences(dataset, time_step=60):
    """Create sequences of data for time series prediction"""
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

def calculate_success(actual, predicted):
    """Calculate success rate based on MAPE"""
    # Handle pandas Series and numpy arrays consistently
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(predicted, pd.Series):
        predicted = predicted.values
        
    # Reshape if needed
    if len(actual.shape) > 1:
        actual = actual.flatten()
    if len(predicted.shape) > 1:
        predicted = predicted.flatten()
        
    # Avoid division by zero
    non_zero = actual != 0
    if np.sum(non_zero) == 0:
        return 0
    
    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
    success = 100 - mape
    return max(0, min(success, 100)) # Clamp between 0 and 100

def add_price_variability(predictions, volatility=0.02, trend_factor=0.3, seed=None):
    """
    Add realistic variability to price predictions.
    Args:
        predictions: Array of predicted prices
        volatility: Base volatility level (0.02 = 2%)
        trend_factor: How much to maintain trend direction (0-1)
        seed: Random seed for reproducibility
    Returns:
        Array of predictions with realistic variations
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Create a copy to avoid modifying the original
    result = predictions.copy()
    
    # Generate random walk component for realistic price movements
    random_factors = np.random.normal(0, volatility, size=len(result))
    
    # Apply exponential smoothing to create trend persistence
    smoothed_factors = np.zeros_like(random_factors)
    smoothed_factors[0] = random_factors[0]
    
    for i in range(1, len(random_factors)):
        smoothed_factors[i] = trend_factor * smoothed_factors[i-1] + (1 - trend_factor) * random_factors[i]
    
    # Apply the smoothed random walk to the predictions
    for i in range(len(result)):
        variation_factor = 1.0 + smoothed_factors[i]
        result[i] = result[i] * variation_factor
        
    return result

def create_prediction_plot(test_dates, actual, pred_lstm):
    """Create an interactive Plotly chart for price predictions with dystopian color scheme"""
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=actual.flatten(),
        mode='lines',
        name='Actual Prices',
        line=dict(color='#ffffff', width=2.5) # White line
    ))
    
    # Add predicted price lines with dystopian colors
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=pred_lstm.flatten(),
        mode='lines',
        name='LSTM Prediction',
        line=dict(color='#00E5FF', width=2, dash='dash') # Neon teal
    ))
    
    # Update layout with dystopian styling
    fig.update_layout(
        title={
            'text': 'Historical Data with Predicted Prices',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': '#0ff8e7'}
        },
        xaxis={
            'title': {
                'text': 'Date',
                'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
            },
            'gridcolor': 'rgba(40, 40, 40, 0.8)',
            'tickfont': {'color': '#8194a9'}
        },
        yaxis={
            'title': {
                'text': 'Price ($)',
                'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
            },
            'gridcolor': 'rgba(40, 40, 40, 0.8)',
            'tickfont': {'color': '#8194a9'}
        },
        hovermode='x unified',
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'family': 'Arial, sans-serif', 'size': 14, 'color': '#a4b8c4'}
        },
        font={'family': 'Arial, sans-serif', 'color': '#a4b8c4'},
        plot_bgcolor='rgba(15, 15, 20, 0.95)',
        paper_bgcolor='rgba(15, 15, 20, 0.95)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_performance_chart(train_success, test_success, models=['LSTM', 'RNN', 'RL']):
    """Generate a Plotly bar chart for model performance comparison with dystopian colors"""
    fig = go.Figure()
    
    # Add bars for train success
    fig.add_trace(go.Bar(
        x=models,
        y=train_success,
        name='Train Success',
        marker_color='#00E5FF', # Neon teal
        text=[f"{val:.2f}%" for val in train_success],
        textposition='auto',
        textfont={'color': '#1a2634'}
    ))
    
    # Add bars for test success
    fig.add_trace(go.Bar(
        x=models,
        y=test_success,
        name='Test Success',
        marker_color='#76FF03', # Toxic green
        text=[f"{val:.2f}%" for val in test_success],
        textposition='auto',
        textfont={'color': '#1a2634'}
    ))
    
    # Update layout with dystopian styling
    fig.update_layout(
        title={
            'text': 'Performance Comparison of Models',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': '#0ff8e7'}
        },
        xaxis={
            'title': {
                'text': 'Models',
                'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
            },
            'tickfont': {'color': '#8194a9'}
        },
        yaxis={
            'title': {
                'text': 'Success Rate (%)',
                'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
            },
            'range': [0, 100],
            'tickfont': {'color': '#8194a9'},
            'gridcolor': 'rgba(40, 40, 40, 0.8)'
        },
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend={'font': {'family': 'Arial, sans-serif', 'size': 14, 'color': '#a4b8c4'}},
        plot_bgcolor='rgba(15, 15, 20, 0.95)',
        paper_bgcolor='rgba(15, 15, 20, 0.95)',
        font={'family': 'Arial, sans-serif', 'color': '#a4b8c4'}
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_comparison_chart(tickers_data):
    """Create a comparison chart for multiple stocks"""
    fig = go.Figure()
    
    # Normalize all prices to the same starting point (100)
    colors = ['#00E5FF', '#76FF03', '#FF1744', '#FFD600', '#F50057'] # Dystopian colors
    
    for i, (ticker, data) in enumerate(tickers_data.items()):
        dates = data['dates']
        prices = data['prices']
        
        # Make sure we have data
        if not prices or len(prices) < 2:
            continue

        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name=ticker,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Update layout with dystopian styling
    fig.update_layout(
        title={
            'text': 'Normalized Price Comparison',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': '#0ff8e7'}
        },
        xaxis={
            'title': {
                'text': 'Date',
                'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
            },
            'gridcolor': 'rgba(40, 40, 40, 0.8)',
            'tickfont': {'color': '#8194a9'}
        },
        yaxis={
            'title': {
                'text': 'Normalized Price (Start = 100)',
                'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
            },
            'gridcolor': 'rgba(40, 40, 40, 0.8)',
            'tickfont': {'color': '#8194a9'}
        },
        hovermode='x unified',
        legend={
            'font': {'family': 'Arial, sans-serif', 'size': 14, 'color': '#a4b8c4'}
        },
        font={'family': 'Arial, sans-serif', 'color': '#a4b8c4'},
        plot_bgcolor='rgba(15, 15, 20, 0.95)',
        paper_bgcolor='rgba(15, 15, 20, 0.95)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# -------------------------------
# Enhanced Sentiment Analysis System
# -------------------------------

class StockSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def scrape_financial_news(self, ticker, limit=10):
        """Scrape financial news from multiple sources for a given ticker"""
        try:
            # Try Finnhub API if available
            news_items = []
            if FINNHUB_AVAILABLE and finnhub_client:
                try:
                    news = finnhub_client.company_news(ticker, 
                                                      _from=(dt.date.today() - dt.timedelta(days=30)).strftime('%Y-%m-%d'),
                                                      to=dt.date.today().strftime('%Y-%m-%d'))
                    
                    # Format the news data
                    for item in news[:limit]:
                        news_items.append({
                            'title': item['headline'],
                            'link': item['url'],
                            'source': item['source'],
                            'date': dt.datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
                        })
                except Exception as e:
                    print(f"Error getting news from Finnhub: {e}")
            
            # If we couldn't get news from Finnhub, fall back to our custom scraper
            if not news_items:
                return self._fallback_scraper(ticker, limit)
                
            return news_items
        except Exception as e:
            print(f"Error in scrape_financial_news: {e}")
            return self._fallback_scraper(ticker, limit)
            
    def _fallback_scraper(self, ticker, limit=10):
        """Fallback scraper for when the API fails"""
        news_items = []
        
        # Function to extract news from Finviz
        def scrape_finviz(ticker):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                url = f'https://finviz.com/quote.ashx?t={ticker}'
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                news_table = soup.find(id='news-table')
                if not news_table:
                    return []
                    
                rows = news_table.findAll('tr')
                
                news_data = []
                for row in rows:
                    title = row.a.text
                    link = row.a['href']
                    date_data = row.td.text.strip().split(' ')
                    
                    if len(date_data) > 1:
                        date = date_data[0]
                        time = date_data[1]
                    else:
                        time = date_data[0]
                    
                    news_data.append({
                        'title': title,
                        'link': link,
                        'source': 'Finviz',
                        'date': dt.datetime.now().strftime('%Y-%m-%d') # Approximate date
                    })
                
                return news_data[:limit]
            except Exception as e:
                print(f"Error scraping Finviz: {e}")
                return []
                
        # Collect news from Finviz
        news_items.extend(scrape_finviz(ticker))
        
        # If no news can be scraped, return an empty list
        # We'll handle this case in get_stock_sentiment
        return news_items[:limit]
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of a given text using both VADER and TextBlob"""
        # VADER sentiment analysis
        vader_scores = self.vader.polarity_scores(text)
        compound_score = vader_scores['compound']
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine both scores (weighted average)
        combined_score = (compound_score * 0.7) + (textblob_polarity * 0.3)
        
        # Determine sentiment label
        if combined_score >= 0.2:
            sentiment = "Positive"
        elif combined_score <= -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "score": combined_score,
            "vader_compound": compound_score,
            "textblob_polarity": textblob_polarity
        }
    
    def get_stock_sentiment(self, ticker, news_limit=15):
        """Get sentiment analysis for a stock based on news articles"""
        # Get news articles
        news_articles = self.scrape_financial_news(ticker, limit=news_limit)
        
        if not news_articles:
            # Instead of returning an error, treat no news as a mixed signal
            current_date = dt.datetime.now().strftime('%Y-%m-%d')
            
            # Create a default mixed signal response
            return {
                "ticker": ticker,
                "price": self._get_stock_price(ticker),
                "market_mood": "Neutral",  # Default to neutral when no news
                "confidence": 0.5,  # Medium confidence
                "average_score": 0.0,  # Neutral score
                "sentiment_counts": {
                    "positive": 0,
                    "neutral": 1,  # Count as one neutral signal
                    "negative": 0
                },
                "price_change_5d": self._get_price_change(ticker),
                "articles": [{
                    "title": f"No recent news found for {ticker}",
                    "sentiment": "Neutral",
                    "score": 0.0,
                    "source": "System",
                    "date": current_date,
                    "link": "#"
                }],
                "mixed_signal": True  # Flag to indicate this is a mixed/no-news signal
            }
            
        # Analyze sentiment for each article
        sentiment_results = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_score = 0
        
        for article in news_articles:
            title = article.get('title', '')
            sentiment_data = self.analyze_sentiment(title)
            
            # Count sentiments
            if sentiment_data['sentiment'] == "Positive":
                positive_count += 1
            elif sentiment_data['sentiment'] == "Negative":
                negative_count += 1
            else:
                neutral_count += 1
                
            total_score += sentiment_data['score']
            
            sentiment_results.append({
                "title": title,
                "sentiment": sentiment_data['sentiment'],
                "score": sentiment_data['score'],
                "source": article.get('source', ''),
                "date": article.get('date', ''),
                "link": article.get('link', '#')
            })
            
        # Calculate overall sentiment
        total_articles = len(sentiment_results)
        average_score = total_score / total_articles if total_articles > 0 else 0
        
        # Determine market mood
        if average_score >= 0.2:
            market_mood = "Bullish"
        elif average_score <= -0.2:
            market_mood = "Bearish"
        else:
            market_mood = "Neutral"
            
        # Calculate confidence based on consensus
        if market_mood == "Bullish":
            confidence = (positive_count / total_articles) if total_articles > 0 else 0
        elif market_mood == "Bearish":
            confidence = (negative_count / total_articles) if total_articles > 0 else 0
        else:
            confidence = (neutral_count / total_articles) if total_articles > 0 else 0
            
        # Ensure confidence is capped at 1.0 (100%)
        confidence = min(round(confidence, 1), 1.0)
        
        return {
            "ticker": ticker,
            "price": self._get_stock_price(ticker),
            "market_mood": market_mood,
            "confidence": confidence,
            "average_score": average_score,
            "sentiment_counts": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count
            },
            "price_change_5d": self._get_price_change(ticker),
            "articles": sentiment_results,
            "mixed_signal": False  # Regular analysis
        }
    
    def _get_stock_price(self, ticker):
        """Get the latest stock price"""
        try:
            stock_data = yf.Ticker(ticker)
            hist = stock_data.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return 0
        except Exception as e:
            print(f"Error getting stock price: {e}")
            return 0
    
    def _get_price_change(self, ticker):
        """Get the 5-day price change percentage"""
        try:
            stock_data = yf.Ticker(ticker)
            hist = stock_data.history(period="5d")
            if not hist.empty and len(hist) > 1:
                price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                return round(price_change, 2)
            return 0
        except Exception as e:
            print(f"Error getting price change: {e}")
            return 0

# -------------------------------
# Company Profiles API
# -------------------------------

import datetime as dt
import yfinance as yf
import pandas as pd
import json
import plotly
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Check if Finnhub is available
try:
    import finnhub
    FINNHUB_API_KEY = "your_finnhub_api_key"  # Replace with actual API key
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    finnhub_client = None


import datetime as dt
import yfinance as yf
import pandas as pd
import json
import plotly
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Check if Finnhub is available
try:
    import finnhub
    FINNHUB_API_KEY = "your_finnhub_api_key"  # Replace with actual API key
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    finnhub_client = None


class CompanyProfiler:
    """Class to handle company profile data retrieval and processing"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = dt.timedelta(days=1)  # Cache for 1 day
        
    def get_company_profile(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company profile data"""
        # Check cache first
        current_time = dt.datetime.now()
        if ticker in self.cache and self.cache_expiry.get(ticker, current_time) > current_time:
            return self.cache[ticker]
            
        try:
            # Try Finnhub first for comprehensive data
            if FINNHUB_AVAILABLE and finnhub_client:
                try:
                    profile = finnhub_client.company_profile2(symbol=ticker)
                    metrics = finnhub_client.company_basic_financials(ticker, 'all')
                    
                    # If we got good data from Finnhub
                    if profile and isinstance(profile, dict) and 'name' in profile:
                        # Create a unified profile object
                        company_data = {
                            'name': profile.get('name', f"{ticker} Inc."),
                            'ticker': ticker,
                            'exchange': profile.get('exchange', 'Unknown'),
                            'industry': profile.get('finnhubIndustry', 'Technology'),
                            'market_cap': profile.get('marketCapitalization', 0) * 1000000,  # Convert to dollars
                            'website': profile.get('weburl', '#'),
                            'logo': profile.get('logo', ''),
                            'country': profile.get('country', 'USA'),
                            'ipo_date': profile.get('ipo', ''),
                            'current_price': self._get_current_price(ticker),
                            'description': profile.get('description', 'No description available.'),
                            'daily_change': 0.0  # Default value
                        }
                        
                        # Add key financial metrics if available
                        if metrics and isinstance(metrics, dict) and 'metric' in metrics:
                            metric_data = metrics['metric']
                            company_data.update({
                                'pe_ratio': metric_data.get('peBasicExclExtraTTM', 0),
                                'eps': metric_data.get('epsBasicExclExtraItemsTTM', 0),
                                'dividend_yield': metric_data.get('dividendYieldIndicatedAnnual', 0),
                                'beta': metric_data.get('beta', 0),
                                '52w_high': metric_data.get('52WeekHigh', 0),
                                '52w_low': metric_data.get('52WeekLow', 0),
                                'revenue': metric_data.get('revenuePerShareTTM', 0),
                                'profit_margin': metric_data.get('netProfitMarginTTM', 0),
                            })
                            
                        # Calculate daily change
                        company_data['daily_change'] = self._calculate_daily_change(ticker)
                            
                        # Cache the result
                        self.cache[ticker] = company_data
                        self.cache_expiry[ticker] = current_time + self.cache_duration
                        
                        return company_data
                except Exception as e:
                    print(f"Error getting Finnhub company data: {e}")
                    # Continue to fallback
            
            # Fallback to Yahoo Finance
            return self._get_yf_profile(ticker)
            
        except Exception as e:
            print(f"Error getting company profile: {e}")
            # Fallback to Yahoo Finance
            return self._get_yf_profile(ticker)
            
    def _get_yf_profile(self, ticker: str) -> Dict[str, Any]:
        """Fallback method to get company profile from Yahoo Finance"""
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            if not info or not isinstance(info, dict) or 'shortName' not in info:
                return self._create_skeleton_profile(ticker)
                
            # Create profile from Yahoo Finance data
            company_data = {
                'name': info.get('shortName', f"{ticker} Inc."),
                'ticker': ticker,
                'exchange': info.get('exchange', 'Unknown'),
                'industry': info.get('industry', 'Technology'),
                'market_cap': info.get('marketCap', 0),
                'website': info.get('website', '#'),
                'logo': '',  # Yahoo doesn't provide logos
                'country': info.get('country', 'USA'),
                'ipo_date': '',  # Not readily available
                'current_price': info.get('currentPrice', self._get_current_price(ticker)),
                'description': info.get('longBusinessSummary', 'No description available.'),
                'pe_ratio': info.get('trailingPE', 0),
                'eps': info.get('trailingEps', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
                'revenue': info.get('revenuePerShare', 0),
                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                'daily_change': 0.0  # Default value
            }
            
            # Calculate daily change
            try:
                if 'previousClose' in info and info['previousClose'] > 0 and company_data['current_price'] > 0:
                    current = company_data['current_price']
                    prev = info['previousClose']
                    daily_change = ((current - prev) / prev) * 100
                    company_data['daily_change'] = round(daily_change, 2)
                else:
                    # Alternative calculation using historical data
                    company_data['daily_change'] = self._calculate_daily_change(ticker)
            except Exception as e:
                print(f"Error calculating daily change: {str(e)}")
                
            # Cache the result
            self.cache[ticker] = company_data
            self.cache_expiry[ticker] = dt.datetime.now() + self.cache_duration
            
            return company_data
            
        except Exception as e:
            print(f"Error getting Yahoo Finance company data: {e}")
            # If all fails, return a skeleton profile
            return self._create_skeleton_profile(ticker)
            
    def _create_skeleton_profile(self, ticker: str) -> Dict[str, Any]:
        """Create a skeleton profile when APIs fail"""
        current_price = self._get_current_price(ticker)
        
        skeleton_profile = {
            'name': f"{ticker} Inc.",
            'ticker': ticker,
            'exchange': 'Unknown',
            'industry': 'Technology',
            'market_cap': 0,
            'website': '#',
            'logo': '',
            'country': 'USA',
            'ipo_date': '',
            'current_price': current_price,
            'description': 'No company information available.',
            'pe_ratio': 0,
            'eps': 0,
            'dividend_yield': 0,
            'beta': 0,
            '52w_high': current_price * 1.2 if current_price else 0,
            '52w_low': current_price * 0.8 if current_price else 0,
            'revenue': 0,
            'profit_margin': 0,
            'daily_change': 0.0
        }
        
        # Cache the skeleton profile for a shorter time (1 hour)
        self.cache[ticker] = skeleton_profile
        self.cache_expiry[ticker] = dt.datetime.now() + dt.timedelta(hours=1)
        
        return skeleton_profile
        
    def _get_current_price(self, ticker: str) -> float:
        """Get the current price for a ticker"""
        try:
            data = yf.download(ticker, period="1d", progress=False)
            if not data.empty and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
            return 0.0
        except Exception as e:
            print(f"Error getting current price for {ticker}: {e}")
            return 0.0
    
    def _calculate_daily_change(self, ticker: str) -> float:
        """Calculate the daily percentage change for a ticker"""
        try:
            df = yf.download(ticker, period="2d", progress=False)
            if not df.empty and len(df) >= 2 and 'Close' in df.columns:
                today_close = df['Close'].iloc[-1]
                yesterday_close = df['Close'].iloc[-2]
                if yesterday_close > 0:  # Avoid division by zero
                    daily_change = ((today_close - yesterday_close) / yesterday_close) * 100
                    return round(daily_change, 2)
            return 0.0
        except Exception as e:
            print(f"Error calculating daily change: {str(e)}")
            return 0.0
            
    def search_companies(self, query: str) -> List[Dict[str, str]]:
        """Search for companies by name or ticker"""
        if not query or len(query.strip()) == 0:
            return []
            
        try:
            # Try using different search methods
            results = []
            
            # Method 1: Try Finnhub API
            if FINNHUB_AVAILABLE and finnhub_client:
                try:
                    finnhub_results = finnhub_client.symbol_lookup(query)
                    if finnhub_results and isinstance(finnhub_results, dict) and 'result' in finnhub_results:
                        for item in finnhub_results['result']:
                            if item.get('type') == 'Common Stock':
                                results.append({
                                    'name': item.get('description', ''),
                                    'ticker': item.get('symbol', ''),
                                    'exchange': item.get('exchange', '')
                                })
                except Exception as e:
                    print(f"Finnhub search error: {e}")
                    
            # If we got results from Finnhub, return them
            if results:
                return results[:10]
                
            # Method 2: Try direct lookup for common tickers
            common_companies = {
                'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN', 'google': 'GOOGL',
                'alphabet': 'GOOGL', 'facebook': 'META', 'meta': 'META', 'netflix': 'NFLX',
                'tesla': 'TSLA', 'nvidia': 'NVDA', 'intel': 'INTC', 'amd': 'AMD',
                'reliance': 'RELIANCE.NS', 'tata': 'TCS.NS', 'infosys': 'INFY',
                'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST', 'disney': 'DIS',
                'coca': 'KO', 'pepsi': 'PEP', 'nike': 'NKE', 'boeing': 'BA',
                'ford': 'F', 'general motors': 'GM', 'chevron': 'CVX', 'exxon': 'XOM'
            }
            
            query_lower = query.lower()
            matched_tickers = []
            
            # Check if query matches any known company names
            for company_name, ticker in common_companies.items():
                if query_lower in company_name or company_name in query_lower:
                    matched_tickers.append(ticker)
                    
            # If the query itself looks like a ticker symbol, add it
            if query.isalpha():
                matched_tickers.append(query.upper())
                
            # Get profiles for matched tickers
            for ticker in matched_tickers:
                try:
                    profile = self.get_company_profile(ticker)
                    if profile:
                        results.append({
                            'name': profile['name'],
                            'ticker': profile['ticker'],
                            'exchange': profile['exchange']
                        })
                except Exception as e:
                    print(f"Error getting profile for {ticker}: {e}")
                    
            # Remove duplicates based on ticker
            unique_results = []
            seen_tickers = set()
            for result in results:
                if result['ticker'] not in seen_tickers:
                    seen_tickers.add(result['ticker'])
                    unique_results.append(result)
                    
            return unique_results[:10]  # Limit to 10 results
        except Exception as e:
            print(f"Error searching companies: {e}")
            return []
    
    def get_stock_history(self, ticker: str, period: str = '1y') -> Dict[str, Any]:
        """Get historical stock data for a ticker"""
        try:
            data = yf.download(ticker, period=period, progress=False)
            if data.empty:
                return {'dates': [], 'prices': []}
                
            # Format the data
            dates = data.index.strftime('%Y-%m-%d').tolist()
            prices = data['Close'].tolist()
            
            return {
                'dates': dates,
                'prices': prices
            }
        except Exception as e:
            print(f"Error getting stock history for {ticker}: {e}")
            return {'dates': [], 'prices': []}
    
    def create_company_chart(self, ticker: str, period: str = '1y') -> str:
        """Create a price chart for a single company using the comparison chart method"""
        # Simply call the comparison chart with a single ticker
        return self.create_comparison_chart([ticker], period)
    
    def create_comparison_chart(self, tickers: List[str], period: str = '1y') -> str:
        """Create a comparison chart for multiple stocks"""
        try:
            if not tickers:
                return json.dumps({})
                
            # Get data for each ticker
            tickers_data = {}
            for ticker in tickers:
                data = self.get_stock_history(ticker, period)
                if data['dates'] and data['prices']:
                    tickers_data[ticker] = data
            
            if not tickers_data:
                return json.dumps({})
                
            # Create figure
            fig = go.Figure()
            
            # Define colors
            colors = ['#00E5FF', '#76FF03', '#FF1744', '#FFD600', '#F50057']  # Dystopian colors
            
            # If we're showing a single stock, use actual prices instead of normalized
            if len(tickers_data) == 1:
                ticker = list(tickers_data.keys())[0]
                data = tickers_data[ticker]
                
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['prices'],
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[0], width=2)
                ))
                
                title_text = f'{ticker} Price History'
                y_axis_title = 'Price'
            else:
                # Normalize all prices to the same starting point (100) for multiple stocks
                for i, (ticker, data) in enumerate(tickers_data.items()):
                    dates = data['dates']
                    prices = data['prices']
                    
                    # Make sure we have data
                    if not prices or len(prices) < 2:
                        continue
                    
                    # Normalize prices
                    start_price = prices[0]
                    if start_price > 0:
                        normalized_prices = [price / start_price * 100 for price in prices]
                        
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=normalized_prices,
                            mode='lines',
                            name=ticker,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                
                title_text = 'Normalized Price Comparison'
                y_axis_title = 'Normalized Price (Start = 100)'
            
            # Update layout with dystopian styling
            fig.update_layout(
                title={
                    'text': title_text,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': '#0ff8e7'}
                },
                xaxis={
                    'title': {
                        'text': 'Date',
                        'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
                    },
                    'gridcolor': 'rgba(40, 40, 40, 0.8)',
                    'tickfont': {'color': '#8194a9'}
                },
                yaxis={
                    'title': {
                        'text': y_axis_title,
                        'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
                    },
                    'gridcolor': 'rgba(40, 40, 40, 0.8)',
                    'tickfont': {'color': '#8194a9'}
                },
                hovermode='x unified',
                legend={
                    'font': {'family': 'Arial, sans-serif', 'size': 14, 'color': '#a4b8c4'}
                },
                font={'family': 'Arial, sans-serif', 'color': '#a4b8c4'},
                plot_bgcolor='rgba(15, 15, 20, 0.95)',
                paper_bgcolor='rgba(15, 15, 20, 0.95)'
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error creating chart: {e}")
            return json.dumps({})

# Initialize the company profiler
company_profiler = CompanyProfiler()

# -------------------------------
# Flask Routes
# -------------------------------

@app.route("/")
def index():
    """Main dashboard page with cosmic design"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    return render_template("index.html", current_date=current_date)


@app.route("/price_predictor", methods=["GET", "POST"])
def price_predictor():
    """Price prediction page using Yahoo Finance data with improved LSTM model"""
    
    # Set non-interactive backend before importing pyplot
    matplotlib.use('Agg')  # Fixes the "main thread is not in main loop" error

    # Helper function to create sequences
    def create_sequences(data, time_step):
        x, y = [], []
        for i in range(time_step, len(data)):
            x.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    # Helper function for metrics visualization
    def create_performance_chart(metrics):
        plt.figure(figsize=(10, 6))
        metrics_keys = list(metrics.keys())
        metrics_values = [float(metrics[key].replace('%', '')) if '%' in metrics[key] else float(metrics[key]) for key in metrics_keys]
        
        bars = plt.bar(metrics_keys, metrics_values, color=['skyblue', 'lightgreen', 'salmon'])
        
        for bar, value in zip(bars, metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, value, ha='center')
        
        plt.title('Model Performance Metrics')
        plt.ylabel('Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()  # Explicitly close the figure
        
        return f"data:image/png;base64,{img_str}"

    # Calculate success rate for model evaluation
    def calculate_success(actual, predicted):
        return 100 - (np.mean(np.abs((actual - predicted) / actual)) * 100)

    # Add price variability for more realistic predictions
    def add_price_variability(preds, volatility=0.03, trend_factor=0.4, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        n = len(preds)
        trend = np.cumsum(np.random.normal(0, 1, n)) * trend_factor * volatility / np.sqrt(n)
        noise = np.random.normal(0, volatility, n)
        return preds * (1 + noise + trend).reshape(-1, 1)
    
    # NEW FUNCTION: Multi-step forecasting by iteratively feeding predictions back as inputs
    def forecast_future_prices(model, last_window, scaler, days_ahead):
        """
        Generate multi-step forecasts by iteratively feeding predictions back as inputs.
        
        Args:
            model: Trained LSTM model
            last_window: The most recent window of prices (scaled)
            scaler: The scaler used to transform the data
            days_ahead: Number of days to forecast
        
        Returns:
            Array of predicted prices (in original scale)
        """
        # Make a copy of the last window to avoid modifying the original
        curr_window = last_window.copy()
        future_predictions = []
        
        # Iterate for each day we want to predict
        for _ in range(days_ahead):
            # Reshape window for prediction (samples, time_steps, features)
            curr_window_reshaped = curr_window.reshape(1, curr_window.shape[0], 1)
            
            # Predict the next day
            next_day = model.predict(curr_window_reshaped, verbose=0)
            
            # Store the prediction (in scaled form for now)
            future_predictions.append(next_day[0, 0])
            
            # Update the window: drop oldest, add newest prediction
            curr_window = np.append(curr_window[1:], next_day[0, 0])
        
        # Convert predictions to original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)
        
        return future_predictions.flatten()

    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    today = dt.date.today()
    ticker_param = request.args.get("ticker")

    # Handle form submission or URL param
    if request.method == "POST" or ticker_param:
        ticker = (request.form.get("ticker", ticker_param) or ticker_param).strip()
        future_date = (request.form.get("future_date") if request.method == "POST" else None) \
                         or (today + dt.timedelta(days=30)).isoformat()

        # ----- Fetch data from Yahoo Finance -----
        try:
            # Use yfinance to get stock data - fetch more data to avoid insufficiency
            start_date = today - dt.timedelta(days=500)  # Get 500 days of data
            end_date = today
            
            # Try different ticker formats for international stocks
            yf_tickers = [ticker]
            # If it seems like a non-US stock without market identifier, try common suffixes
            if '.' not in ticker and ':' not in ticker:
                # Add common international exchange suffixes
                yf_tickers.extend([f"{ticker}.NS", f"{ticker}.BO", f"{ticker}.L", f"{ticker}.TO"])
            
            # Try each ticker format
            df = None
            successful_ticker = None
            
            for yf_ticker in yf_tickers:
                try:
                    # Use multi_level_index=False to avoid the multi-index issue
                    df_tmp = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, multi_level_index=False)
                    if not df_tmp.empty and len(df_tmp) > 60:  # Ensure we have enough data
                        df = pd.DataFrame(df_tmp['Close'])
                        successful_ticker = yf_ticker
                        break
                except Exception:
                    continue
            
            if df is None or df.empty:
                raise ValueError(f"No data available for ticker {ticker} or its variations")
            
            # Data source info for display
            data_source = f"Yahoo Finance (as {successful_ticker})"
                
        except Exception as e:
            return render_template(
                "price_predictor.html",
                error=f"Failed to retrieve data from Yahoo Finance: {str(e)}",
                current_date=current_date,
                today=today
            )

        # ----- Preprocess for modeling -----
        data_vals = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_vals)

        # Adaptive time_step based on available data
        if len(scaled_data) < 150:
            time_step = min(50, max(30, int(len(scaled_data) * 0.3)))  # Use between 30-50 days or 30% of data
        else:
            time_step = 60  # Default time step for sufficient data

        # Check if we have enough data after adjustment
        if len(scaled_data) <= time_step:
            return render_template(
                "price_predictor.html",
                error=f"Not enough data points for prediction. Need at least {time_step+1} days, but only have {len(scaled_data)}.",
                current_date=current_date,
                today=today
            )

        # Using 70% for training as recommended in the notebook
        train_size = int(len(scaled_data) * 0.7)

        # Create sequences for training
        x_train, y_train = create_sequences(scaled_data[:train_size], time_step)
        x_train = x_train.reshape(-1, time_step, 1)
        
        # Create sequences for testing
        x_test, y_test = create_sequences(scaled_data[train_size - time_step:], time_step)
        x_test = x_test.reshape(-1, time_step, 1)

        # Adapt epochs based on data size
        epochs = min(100, max(50, int(len(scaled_data) / 10)))  # Scale epochs with data size
        training_progress = []

        # ----- Improved LSTM Model using Input layer (fixes warning) -----
        lstm_model = Sequential()
        lstm_model.add(Input(shape=(time_step, 1)))  # Proper input layer
        lstm_model.add(LSTM(50, activation='relu', return_sequences=True))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(60, activation='relu', return_sequences=True))
        lstm_model.add(Dropout(0.3))
        lstm_model.add(LSTM(80, activation='relu', return_sequences=True))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(LSTM(120, activation='relu'))
        lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(1))
        
        lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        # Training with progress tracking
        for epoch in range(epochs):
            lstm_model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
            training_progress.append({"epoch": epoch + 1, "progress": ((epoch + 1)/epochs)*100})

        # ----- Make predictions -----
        pred_train = lstm_model.predict(x_train)
        pred_test = lstm_model.predict(x_test)
        
        # Inverse transform predictions
        pred_train = scaler.inverse_transform(pred_train)
        pred_test = scaler.inverse_transform(pred_test)
        actual_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # ----- Model Evaluation -----
        mae = mean_absolute_error(actual_test, pred_test)
        mae_percentage = (mae / np.mean(actual_test)) * 100
        r2 = r2_score(actual_test, pred_test)
        
        # Calculate success rate (compatibility with original template)
        train_success = calculate_success(actual_train, pred_train)
        test_success = calculate_success(actual_test, pred_test)

        # ----- Future Prediction with Multi-Step Forecasting -----
        # Calculate days between current date and target future date
        future_date_obj = dt.datetime.fromisoformat(future_date)
        days_ahead = (future_date_obj.date() - today).days
        
        # Make sure we're forecasting at least 1 day ahead
        days_ahead = max(1, days_ahead)
        
        # Get the last window of data for forecasting
        last_window = scaled_data[-time_step:]
        
        # Generate multi-step forecast
        future_predictions = forecast_future_prices(lstm_model, last_window, scaler, days_ahead)
        
        # The final day's prediction is what we care about for the target date
        future_pred = future_predictions[-1]
        
        # Add realistic variability
        future_pred = future_pred * (1 + random.uniform(-0.03, 0.03))

        # ----- Create charts -----
        test_dates = df.index[train_size:train_size + len(pred_test)]
        price_plot = create_prediction_plot(test_dates, actual_test, pred_test)
        
        performance_metrics = {
            "MAE": f"{mae:.2f}",
            "MAE %": f"{mae_percentage:.2f}%",
            "R² Score": f"{r2:.4f}"
        }
        performance_plot = create_performance_chart(performance_metrics)
        
        # ----- Prepare results structure for template compatibility -----
        models = [
            {"name": "LSTM", "train_success": f"{train_success:.2f}", "test_success": f"{test_success:.2f}"}
        ]
        
        model_metrics = {
            "mae": f"{mae:.2f}",
            "mae_percentage": f"{mae_percentage:.2f}%",
            "r2_score": f"{r2:.4f}"
        }
        
        # Get company profile and fix handling for dict instead of DataFrame
        raw_company_info = company_profiler.get_company_profile(ticker)
        
        # Initialize company_info dict
        company_info = {}
        
        # Safely extract current price from DataFrame for default values
        current_price = 0.0
        try:
            current_price = float(df['Close'].iloc[-1])
        except (IndexError, ValueError, AttributeError):
            pass
            
        # Handle company info based on its type
        if isinstance(raw_company_info, dict):
            # It's already a dictionary
            for key, value in raw_company_info.items():
                # Handle pandas Series or scalar values
                if hasattr(value, 'iloc'):
                    try:
                        # Extract scalar from Series
                        if key in ['daily_change', 'current_price']:
                            company_info[key] = float(value.iloc[0])
                        else:
                            company_info[key] = value.iloc[0]
                    except (IndexError, ValueError, TypeError):
                        # Fallback values
                        company_info[key] = 0.0 if key in ['daily_change', 'current_price'] else "N/A"
                else:
                    # Direct scalar value handling
                    try:
                        if key in ['daily_change', 'current_price']:
                            company_info[key] = float(value)
                        else:
                            company_info[key] = value
                    except (ValueError, TypeError):
                        company_info[key] = 0.0 if key in ['daily_change', 'current_price'] else "N/A"
        elif hasattr(raw_company_info, 'empty'):
            # It's a DataFrame
            if not raw_company_info.empty:
                for column in raw_company_info.columns:
                    try:
                        if column in ['daily_change', 'current_price']:
                            company_info[column] = float(raw_company_info[column].iloc[0])
                        else:
                            company_info[column] = raw_company_info[column].iloc[0]
                    except (IndexError, ValueError, TypeError):
                        company_info[column] = 0.0 if column in ['daily_change', 'current_price'] else "N/A"
        
        # If company_info is still empty, use defaults
        if not company_info:
            company_info = {
                'name': ticker.upper(),
                'current_price': current_price,
                'daily_change': 0.0,
                'market_cap': 'N/A',
                'volume': 'N/A',
                'description': 'No company description available'
            }
        
        # Ensure current_price is always available
        if 'current_price' not in company_info or not company_info['current_price']:
            company_info['current_price'] = current_price
        
        # Make sure current_price is float for template compatibility
        try:
            company_info['current_price'] = float(company_info['current_price'])
        except (ValueError, TypeError):
            company_info['current_price'] = current_price
            
        # Make sure daily_change is float for template compatibility
        if 'daily_change' not in company_info:
            company_info['daily_change'] = 0.0
        try:
            company_info['daily_change'] = float(company_info['daily_change'])
        except (ValueError, TypeError):
            company_info['daily_change'] = 0.0
        
        # Format results for template compatibility
        results = {
            "ticker": ticker.upper(),
            "future_date": future_date,
            "future_pred_lstm": f"{future_pred:.2f}",  # Keep original key for template compatibility
            "future_pred_rnn": f"{future_pred:.2f}",   # For template compatibility
            "future_pred_rl": f"{future_pred:.2f}",    # For template compatibility
            "models": models,
            "model_metrics": model_metrics,
            "training_progress": training_progress,
            "company_info": company_info,
            "data_source": data_source
        }

        return render_template(
            "price_predictor.html",
            results=results,
            price_plot=price_plot,
            performance_plot=performance_plot,
            current_date=current_date,
            today=today
        )

    # GET: default 30-day future date
    default_date = (today + dt.timedelta(days=30)).isoformat()
    return render_template(
        "price_predictor.html",
        default_date=default_date,
        current_date=current_date,
        ticker=ticker_param,
        today=today
    )








@app.route("/sentiment", methods=["GET", "POST"])
def sentiment_analyzer():
    """Sentiment analysis page with cosmic design"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    # Check if ticker is provided in query parameters
    ticker_param = request.args.get('ticker')
    
    if request.method == "POST" or ticker_param:
        # Get ticker from form or query parameter
        ticker = request.form.get("ticker", ticker_param).strip() if request.method == "POST" else ticker_param
        
        # Initialize the sentiment analyzer
        analyzer = StockSentimentAnalyzer()
        
        try:
            # Get sentiment data with the improved analyzer
            sentiment_results = analyzer.get_stock_sentiment(ticker)
            
            if "error" in sentiment_results:
                return render_template("sentiment.html", error=sentiment_results["error"], current_date=current_date)
                
            # Get company profile for additional context
            company_info = company_profiler.get_company_profile(ticker)
            sentiment_results["company_info"] = company_info
            
            return render_template("sentiment.html", results=sentiment_results, current_date=current_date)
            
        except Exception as e:
            return render_template("sentiment.html", error=f"Error analyzing sentiment: {str(e)}", current_date=current_date)
            
    return render_template("sentiment.html", current_date=current_date)

@app.route("/portfolio", methods=["GET", "POST"])
def portfolio_optimizer():
    """Portfolio optimization page with cosmic design"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    if request.method == "POST":
        # Get tickers from form
        tickers_input = request.form["tickers"].strip()
        tickers = [t.strip() for t in tickers_input.split(',')]
        
        # Get risk preference from form with validation
        risk_preference = request.form.get("risk_preference", "moderate")
        
        # Validate risk preference is one of the allowed values
        if risk_preference not in ["low", "moderate", "high"]:
            risk_preference = "moderate"  # Default to moderate if invalid value
        
        try:
            # Initialize optimizer
            optimizer = PortfolioOptimizer()
            
            # Optimize portfolio
            results = optimizer.optimize_portfolio(tickers, risk_preference=risk_preference)
            
            if results is None:
                return render_template("portfolio.html", 
                                      error="No data found for the given tickers.", 
                                      current_date=current_date,
                                      risk_options=["low", "moderate", "high"],
                                      selected_risk=risk_preference)
            
            # Get company profiles for all tickers
            for asset in results["portfolio_data"]:
                ticker = asset["ticker"]
                asset["company_info"] = company_profiler.get_company_profile(ticker)
            
            return render_template("portfolio.html", 
                                  results=results, 
                                  current_date=current_date,
                                  risk_options=["low", "moderate", "high"],
                                  selected_risk=risk_preference)
            
        except Exception as e:
            return render_template("portfolio.html", 
                                  error=f"Error optimizing portfolio: {str(e)}", 
                                  current_date=current_date,
                                  risk_options=["low", "moderate", "high"],
                                  selected_risk=risk_preference)
    
    # For GET requests, just render the form with default options
    return render_template("portfolio.html", 
                          current_date=current_date,
                          risk_options=["low", "moderate", "high"],
                          selected_risk="moderate")

@app.route("/ticker_search", methods=["GET", "POST"])
def ticker_search():
    """Ticker search functionality to find company stock symbols"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    if request.method == "POST":
        company_name = request.form["company_name"].strip()
        
        try:
            # Search for companies based on input
            search_results = company_profiler.search_companies(company_name)
            
            if not search_results:
                return render_template("ticker_search.html", error="No companies found matching your search.", current_date=current_date)
                
            # Get more detailed information for each result
            results = []
            for company in search_results:
                try:
                    ticker = company["ticker"]
                    profile = company_profiler.get_company_profile(ticker)
                    results.append(profile)
                except Exception as e:
                    print(f"Error getting profile for {company['ticker']}: {e}")
                    # Add a basic profile if unable to get full details
                    results.append({
                        'name': company.get('name', company['ticker']),
                        'ticker': company['ticker'],
                        'exchange': company.get('exchange', 'Unknown'),
                        'industry': 'Unknown',
                        'market_cap': 0,
                        'current_price': 0
                    })
                
            return render_template("ticker_search.html", results=results, current_date=current_date)
            
        except Exception as e:
            return render_template("ticker_search.html", error=f"Error searching for ticker: {str(e)}", current_date=current_date)
            
    return render_template("ticker_search.html", current_date=current_date)

@app.route("/company_profile")
def company_profile():
    """Detailed company profile page"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    ticker = request.args.get('ticker')
    
    if not ticker:
        return render_template("company_profile.html", error="No ticker symbol specified.", current_date=current_date)
        
    try:
        # Get complete company profile
        profile = company_profiler.get_company_profile(ticker)
        
        # Get stock price history for charts
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=365)
        
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            return render_template("company_profile.html", error=f"No price data found for {ticker}.", current_date=current_date)
            
        # Create price history chart
        price_history = go.Figure()
        price_history.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#00E5FF', width=2)
        ))
        
        price_history.update_layout(
            title={
                'text': f'{ticker} Price History',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'family': 'Arial, sans-serif', 'size': 24, 'color': '#0ff8e7'}
            },
            xaxis={
                'title': {
                    'text': 'Date',
                    'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
                },
                'gridcolor': 'rgba(40, 40, 40, 0.8)',
                'tickfont': {'color': '#8194a9'}
            },
            yaxis={
                'title': {
                    'text': 'Price ($)',
                    'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': '#8194a9'}
                },
                'gridcolor': 'rgba(40, 40, 40, 0.8)',
                'tickfont': {'color': '#8194a9'}
            },
            plot_bgcolor='rgba(15, 15, 20, 0.95)',
            paper_bgcolor='rgba(15, 15, 20, 0.95)',
            font={'family': 'Arial, sans-serif', 'color': '#a4b8c4'}
        )
        
        price_chart = json.dumps(price_history, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get sentiment data as well
        analyzer = StockSentimentAnalyzer()
        sentiment_data = analyzer.get_stock_sentiment(ticker)
        
        return render_template(
            "company_profile.html",
            profile=profile,
            price_chart=price_chart,
            sentiment_data=sentiment_data,
            current_date=current_date
        )
        
    except Exception as e:
        return render_template("company_profile.html", error=f"Error getting company profile: {str(e)}", current_date=current_date)

@app.route("/compare", methods=["GET", "POST"])
def compare():
    """Stock comparison tool"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    if request.method == "POST":
        tickers_input = request.form["tickers"].strip()
        tickers = [t.strip() for t in tickers_input.split(',')]
        period = request.form.get("period", "1y")
        
        try:
            # Validate tickers
            if len(tickers) < 2:
                return render_template("compare.html", error="Please enter at least two ticker symbols separated by commas.", current_date=current_date)
                
            if len(tickers) > 5:
                return render_template("compare.html", error="Please enter at most 5 ticker symbols for comparison.", current_date=current_date)
                
            # Get data for all tickers
            end_date = dt.date.today()
            start_date = {
                "1m": end_date - dt.timedelta(days=30),
                "3m": end_date - dt.timedelta(days=90),
                "6m": end_date - dt.timedelta(days=180),
                "1y": end_date - dt.timedelta(days=365),
                "3y": end_date - dt.timedelta(days=3*365),
                "5y": end_date - dt.timedelta(days=5*365)
            }.get(period, end_date - dt.timedelta(days=365))
            
            # Collect data for each ticker separately to avoid dimension issues
            tickers_data = {}
            company_profiles = {}
            
            for ticker in tickers:
                try:
                    # Download data for this single ticker - specify auto_adjust=True explicitly
                    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                    
                    if df.empty or len(df) < 5:  # Require at least 5 data points
                        continue
                        
                    # Ensure 'Close' column exists
                    if 'Close' not in df.columns:
                        continue
                        
                    # Convert index to string dates for JSON serialization
                    dates = df.index.strftime('%Y-%m-%d').tolist()
                    
                    # Get close prices as individual scalars, not arrays
                    close_prices = []
                    for idx in range(len(df)):
                        close_prices.append(float(df['Close'].iloc[idx]))
                    
                    # Calculate returns from close prices 
                    returns = []
                    for i in range(len(close_prices)):
                        if i == 0:
                            returns.append(0.0)  # First return is 0
                        else:
                            prev_price = close_prices[i-1]
                            if prev_price > 0:
                                ret = (close_prices[i] / prev_price) - 1
                                returns.append(float(ret))
                            else:
                                returns.append(0.0)
                    
                    # Get volume data similarly
                    volume = []
                    if 'Volume' in df.columns:
                        for idx in range(len(df)):
                            volume.append(float(df['Volume'].iloc[idx]))
                    
                    # Store data as simple lists to avoid dimensionality issues
                    tickers_data[ticker] = {
                        'dates': dates,
                        'prices': close_prices,
                        'returns': returns,
                        'volume': volume
                    }
                    
                    # Get company profile for additional context
                    try:
                        company_profiles[ticker] = company_profiler.get_company_profile(ticker)
                    except Exception as profile_error:
                        print(f"Error getting profile for {ticker}: {profile_error}")
                        # Create a basic profile if detailed one fails
                        company_profiles[ticker] = {
                            'name': ticker,
                            'ticker': ticker,
                            'exchange': 'Unknown',
                            'industry': 'Unknown',
                            'market_cap': 0,
                            'current_price': float(df['Close'].iloc[-1]) if not df.empty else 0
                        }
                        
                except Exception as e:
                    print(f"Error downloading data for {ticker}: {e}")
                    continue
                    
            if not tickers_data or len(tickers_data) < 2:
                return render_template("compare.html", error="Could not find sufficient data for at least two of the provided tickers.", current_date=current_date)
                
            # Generate comparison chart
            try:
                price_chart = create_comparison_chart(tickers_data)
            except Exception as chart_error:
                print(f"Error creating comparison chart: {chart_error}")
                price_chart = None
                
            # Calculate correlation matrix - ensure data is properly aligned
            ticker_list = list(tickers_data.keys())
            correlation_matrix = [[1.0 for _ in ticker_list] for _ in ticker_list]  # Default to 1.0 (perfect correlation)
            
            try:
                # Create a DataFrame with aligned dates
                all_returns_data = {}
                all_dates = set()
                
                # Collect all unique dates
                for ticker, data in tickers_data.items():
                    all_dates.update(data['dates'])
                
                # Sort dates
                all_dates = sorted(list(all_dates))
                
                # Initialize empty return series for all tickers across all dates
                for ticker in ticker_list:
                    all_returns_data[ticker] = {date: None for date in all_dates}
                
                # Fill in the returns data where available
                for ticker, data in tickers_data.items():
                    for i, date in enumerate(data['dates']):
                        if i < len(data['returns']):
                            all_returns_data[ticker][date] = data['returns'][i]
                
                # Create a properly aligned DataFrame
                all_returns = pd.DataFrame(index=all_dates)
                for ticker in ticker_list:
                    all_returns[ticker] = [all_returns_data[ticker][date] for date in all_dates]
                
                # Drop any rows with missing data
                all_returns = all_returns.dropna()
                
                # Calculate correlation matrix if we have data
                if not all_returns.empty and all_returns.shape[1] >= 2:
                    corr_matrix = all_returns.corr().values
                    
                    # Convert to list of lists, ensuring proper type conversion
                    correlation_matrix = []
                    for i in range(corr_matrix.shape[0]):
                        correlation_matrix.append([float(corr_matrix[i,j]) for j in range(corr_matrix.shape[1])])
                        
            except Exception as corr_error:
                print(f"Error calculating correlation matrix: {corr_error}")
                # Fallback to identity matrix if correlation calculation fails
                correlation_matrix = [[1.0 if i == j else 0.0 for j in range(len(ticker_list))] for i in range(len(ticker_list))]
                
            # Calculate key statistics
            stats = {}
            for ticker, data in tickers_data.items():
                prices = data['prices']  # This is now a list of floats
                returns = data['returns'][1:] if len(data['returns']) > 1 else []  # Skip the first NaN
                
                if not prices or len(prices) < 2:
                    continue
                    
                # Calculate max drawdown safely
                max_drawdown = 0
                try:
                    peak = prices[0]
                    for price in prices:
                        if price > peak:
                            peak = price
                        drawdown = (peak - price) / peak * 100 if peak > 0 else 0
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                except Exception as dd_error:
                    print(f"Error calculating drawdown: {dd_error}")
                    
                stats[ticker] = {
                    'start_price': prices[0],
                    'end_price': prices[-1],
                    'change_pct': ((prices[-1] / prices[0]) - 1) * 100 if prices[0] > 0 else 0,
                    'volatility': float(np.std(returns) * np.sqrt(252) * 100) if len(returns) > 0 else 0,
                    'max_drawdown': max_drawdown
                }
                
            # Create the results object
            results = {
                'tickers': ticker_list,
                'period': period,
                'company_profiles': company_profiles,
                'statistics': stats,
                'correlation_matrix': correlation_matrix
            }
            
            return render_template(
                "compare.html",
                results=results,
                price_chart=price_chart,
                current_date=current_date
            )
            
        except Exception as e:
            return render_template("compare.html", error=f"Error comparing stocks: {str(e)}", current_date=current_date)
            
    return render_template("compare.html", current_date=current_date)
# -------------------------------
# User Account Routes
# -------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            # Login successful
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template("login.html", current_date=current_date)

@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration page"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        try:
            # Validation
            if password != confirm_password:
                flash("Passwords do not match.", "error")
                return render_template("register.html", current_date=current_date)
                
            # Check if username or email already exists
            if User.query.filter_by(username=username).first():
                flash("Username already exists.", "error")
                return render_template("register.html", current_date=current_date)
                
            if User.query.filter_by(email=email).first():
                flash("Email already exists.", "error")
                return render_template("register.html", current_date=current_date)
                
            # Validate email format
            try:
                valid = validate_email(email)
                email = valid.email
            except EmailNotValidError:
                flash("Invalid email address.", "error")
                return render_template("register.html", current_date=current_date)
                
            # Create new user
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
            
        except Exception as e:
            flash(f"Registration error: {str(e)}", "error")
            
    return render_template("register.html", current_date=current_date)

@app.route("/logout")
def logout():
    """User logout"""
    session.pop('user_id', None)
    session.pop('username', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    """Forgot password page"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    if request.method == "POST":
        email = request.form["email"]
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate reset token
            token = user.generate_reset_token()
            
            # Send reset email
            reset_url = url_for('reset_password', token=token, user_id=user.id, _external=True)
            
            # Create email message
            msg = Message("Cosmic Finance Password Reset", recipients=[email])
            msg.body = f"""Hello {user.username},
            
You recently requested to reset your password. Please use the link below to reset it:

{reset_url}

This link will expire in 24 hours.

If you did not request a password reset, please ignore this email.

Regards,

Cosmic Finance Team
"""
            
            try:
                mail.send(msg)
                flash("Password reset instructions sent to your email.", "success")
            except Exception as e:
                print(f"Error sending email: {e}")
                flash("Unable to send email. Please try again later.", "error")
        else:
            # Don't reveal if email exists or not for security
            flash("If an account with that email exists, password reset instructions have been sent.", "info")
            
        return redirect(url_for('login'))
        
    return render_template("forgot_password.html", current_date=current_date)

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """Reset password page"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    # Get user ID from query parameter
    user_id = request.args.get('user_id')
    
    if not user_id:
        flash("Invalid reset link.", "error")
        return redirect(url_for('login'))
        
    # Find user
    user = User.query.get(user_id)
    
    if not user or not user.verify_reset_token(token):
        flash("The password reset link is invalid or has expired.", "error")
        return redirect(url_for('login'))
        
    if request.method == "POST":
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("reset_password.html", token=token, user_id=user_id, current_date=current_date)
            
        # Update password
        user.set_password(password)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        
        flash("Your password has been updated. Please log in with your new password.", "success")
        return redirect(url_for('login'))
        
    return render_template("reset_password.html", token=token, user_id=user_id, current_date=current_date)

@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard with their saved portfolios and favorites"""
    current_date = dt.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')
    
    # Get user information
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user:
        flash("User not found.", "error")
        return redirect(url_for('logout'))
        
    # Get user's favorites
    favorites = Favorite.query.filter_by(user_id=user_id).all()
    favorite_data = []
    
    for favorite in favorites:
        try:
            # Get current price and basic info
            ticker_data = yf.Ticker(favorite.ticker)
            hist = ticker_data.history(period="2d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0] if len(hist) > 1 else current_price
                price_change = ((current_price - prev_price) / prev_price) * 100
            else:
                current_price = 0
                price_change = 0
                
            favorite_data.append({
                'id': favorite.id,
                'ticker': favorite.ticker,
                'notes': favorite.notes,
                'current_price': current_price,
                'price_change': price_change,
                'created_at': favorite.created_at
            })
        except Exception as e:
            print(f"Error getting favorite data for {favorite.ticker}: {e}")
            favorite_data.append({
                'id': favorite.id,
                'ticker': favorite.ticker,
                'notes': favorite.notes,
                'current_price': 0,
                'price_change': 0,
                'created_at': favorite.created_at
            })
    
    # Get user's portfolios
    portfolios = Portfolio.query.filter_by(user_id=user_id).all()
    portfolio_data = []
    
    for portfolio in portfolios:
        holdings = Holding.query.filter_by(portfolio_id=portfolio.id).all()
        total_value = 0
        total_cost = 0
        
        for holding in holdings:
            try:
                ticker_data = yf.Ticker(holding.ticker)
                hist = ticker_data.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    current_value = current_price * holding.quantity
                    cost_basis = holding.purchase_price * holding.quantity
                else:
                    current_price = 0
                    current_value = 0
                    cost_basis = 0
                
                total_value += current_value
                total_cost += cost_basis
            except Exception as e:
                print(f"Error calculating portfolio value for {holding.ticker}: {e}")
        
        # Calculate total return
        if total_cost > 0:
            total_return = ((total_value - total_cost) / total_cost) * 100
        else:
            total_return = 0
            
        portfolio_data.append({
            'id': portfolio.id,
            'name': portfolio.name,
            'description': portfolio.description,
            'total_value': total_value,
            'total_return': total_return,
            'holdings_count': len(holdings),
            'created_at': portfolio.created_at
        })
        
    return render_template(
        "dashboard.html",
        user=user,
        favorites=favorite_data,
        portfolios=portfolio_data,
        current_date=current_date
    )

@app.route('/create_db')
def create_database():
    """Create the database tables"""
    try:
        db.create_all()
        return "Database created successfully"
    except Exception as e:
        return f"Error creating database: {str(e)}"



if __name__ == "__main__":
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Run the app
    app.run(debug=True)
