# portfolio_zerodha.py
import os
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv
from kiteconnect import KiteConnect, exceptions as kite_exceptions
from langchain_core.tools import tool
from typing import Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Where we'll persist the access token
SESSION_FILE = "zerodha_session.json"
TOKEN_FILE = "zerodha_request_token.txt"
load_dotenv()

class TokenHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        if 'request_token' in query_components:
            token = query_components['request_token'][0]
            # Save token to file
            with open(TOKEN_FILE, 'w') as f:
                f.write(token)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Request token received and saved. You can close this window.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid request")

def ensure_valid_session():
    """Ensures a valid session exists by handling the request token process if needed."""
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Missing ZERODHA_API_KEY or ZERODHA_API_SECRET in environment.")

    kite = KiteConnect(api_key=api_key)
    
    # Check if we have a valid session
    access_token = None
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            saved = json.load(f)
            access_token = saved.get("access_token")
    
    if access_token:
        kite.set_access_token(access_token)
        try:
            kite.profile()
            return  # Valid session exists
        except kite_exceptions.TokenException:
            pass  # Session is invalid, continue to get new token
    
    # No valid session, need to get new token
    login_url = kite.login_url()
    
    # Start a local server to receive the request token
    server = HTTPServer(('localhost', 5000), TokenHandler)
    
    # Open the login URL in the default browser
    webbrowser.open(login_url)
    
    # Wait for the request token (with timeout)
    import time
    start_time = time.time()
    while not os.path.exists(TOKEN_FILE) and time.time() - start_time < 300:  # 5 minute timeout
        server.handle_request()
    
    if not os.path.exists(TOKEN_FILE):
        raise TimeoutError("Failed to get request token within timeout period")
    
    # Read the saved token
    with open(TOKEN_FILE, 'r') as f:
        request_token = f.read().strip()
    
    # Generate session and save access token
    sess = kite.generate_session(request_token, api_secret=api_secret)
    access_token = sess["access_token"]
    with open(SESSION_FILE, "w") as f:
        json.dump({"access_token": access_token}, f, indent=2)
    
    # Clean up the request token file
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)

@tool
def get_zerodha_portfolio(input: str = "") -> dict:
    """
    Fetches and returns your Zerodha portfolio, handling session persistence.

    Args:
        None

    Returns:
        dict with keys 'holdings', 'net_positions', 'day_positions'

    Raises:
        ValueError: if API credentials are missing or request_token is needed but not provided
    """
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Missing ZERODHA_API_KEY or ZERODHA_API_SECRET in environment.")

    # Ensure we have a valid session
    ensure_valid_session()
    
    # Now we can safely use the saved session
    with open(SESSION_FILE, "r") as f:
        saved = json.load(f)
        access_token = saved["access_token"]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    holdings = kite.holdings()
    positions_all = kite.positions()
    return {
        "holdings": holdings,
        "net_positions": positions_all.get("net", []),
        "day_positions": positions_all.get("day", [])
    }

class PortfolioInput(BaseModel):
    query: str = Field(description="The query about portfolio analysis")

class PortfolioTool(BaseTool):
    name: str = "portfolio_analyzer"
    description: str = "A tool that analyzes investment portfolios and provides recommendations"
    args_schema: type[BaseModel] = PortfolioInput

    def _run(self, query: str) -> Dict[str, Any]:
        """
        Analyze a portfolio and provide recommendations.
        This is a mock implementation that simulates portfolio data.
        In a real implementation, this would connect to Zerodha's API.
        """
        try:
            # Mock portfolio data
            portfolio = {
                'AAPL': {'quantity': 10, 'avg_price': 150.0},
                'MSFT': {'quantity': 5, 'avg_price': 250.0},
                'GOOGL': {'quantity': 3, 'avg_price': 2800.0},
                'AMZN': {'quantity': 2, 'avg_price': 3300.0},
                'TSLA': {'quantity': 1, 'avg_price': 800.0}
            }
            
            # Get current prices
            current_prices = {}
            total_value = 0
            for symbol in portfolio:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        current_prices[symbol] = current_price
                        total_value += current_price * portfolio[symbol]['quantity']
                except Exception as e:
                    print(f"Error getting price for {symbol}: {str(e)}")
            
            # Calculate holdings
            holdings = []
            for symbol, data in portfolio.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    quantity = data['quantity']
                    value = current_price * quantity
                    holdings.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'current_price': current_price,
                        'value': value,
                        'percentage': (value / total_value) * 100
                    })
            
            # Sort holdings by value
            holdings.sort(key=lambda x: x['value'], reverse=True)
            
            # Format top holdings
            top_holdings = "\n".join([
                f"- {h['symbol']}: {h['quantity']} shares (${h['value']:.2f}, {h['percentage']:.1f}%)"
                for h in holdings[:5]
            ])
            
            # Generate recommendations
            recommendations = []
            
            # Check for overconcentration
            if holdings[0]['percentage'] > 20:
                recommendations.append(f"Consider reducing exposure to {holdings[0]['symbol']} as it represents {holdings[0]['percentage']:.1f}% of your portfolio")
            
            # Check for diversification
            if len(holdings) < 5:
                recommendations.append("Consider adding more stocks to diversify your portfolio")
            
            # Check for sector concentration
            tech_stocks = sum(1 for h in holdings if h['symbol'] in ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
            if tech_stocks > 3:
                recommendations.append("Your portfolio is heavily concentrated in technology stocks. Consider adding stocks from other sectors")
            
            # Format recommendations
            recommendations_text = "\n".join([f"- {rec}" for rec in recommendations])
            
            return {
                'total_value': f"{total_value:.2f}",
                'num_holdings': len(holdings),
                'stocks_percentage': 100,  # Mock data
                'bonds_percentage': 0,     # Mock data
                'cash_percentage': 0,      # Mock data
                'top_holdings': top_holdings,
                'recommendations': recommendations_text
            }
            
        except Exception as e:
            return {"error": f"Error analyzing portfolio: {str(e)}"}

    async def _arun(self, query: str) -> Dict[str, Any]:
        return self._run(query)

# Create the tool instance
portfolio_analyzer = PortfolioTool()

