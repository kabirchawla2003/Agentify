from typing import Dict, List
import os
from dotenv import load_dotenv
import yfinance as yf
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Common company names to symbols mapping
COMPANY_SYMBOLS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    # Indian stocks
    "reliance": "RELIANCE.NS",
    "tata motors": "TATAMOTORS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "tcs": "TCS.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "wipro": "WIPRO.NS"
}

def format_stock_data(data) -> List[Dict]:
    """Format stock data into a list of dictionaries."""
    formatted_data = []
    for index, row in data.iterrows():
        formatted_data.append({
            "Date": index.strftime("%Y-%m-%d"),
            "Open": str(row["Open"]),
            "High": str(row["High"]),
            "Low": str(row["Low"]),
            "Close": str(row["Close"]),
            "Volume": str(row["Volume"])
        })
    return formatted_data

def get_symbol_for_company(company_name: str) -> str:
    """Get the stock symbol for a company name."""
    # Clean the company name
    company_name = company_name.lower().strip()
    # Remove common suffixes
    for suffix in [" inc", " inc.", " corporation", " corp", " corp.", " ltd", " ltd."]:
        company_name = company_name.replace(suffix, "")
    
    # First check our predefined mapping
    for key in COMPANY_SYMBOLS:
        if key in company_name:
            return COMPANY_SYMBOLS[key]
    
    # If not found in mapping, try with .NS suffix for Indian stocks
    if any(indian_word in company_name for indian_word in ["india", "indian", "bombay", "mumbai"]):
        return f"{company_name.split()[0].upper()}.NS"
    
    # Return the uppercase version of the first word as a fallback
    return company_name.split()[0].upper()

@tool
def get_stock_data(company_name: str) -> Dict:
    """
    Fetches stock data for the given company using yfinance.

    Args:
        company_name: Name of the company or stock symbol (e.g., "Apple" or "AAPL")

    Returns:
        A dictionary containing the last 5 weeks of stock data and last 5 years of the stock data.
    """
    try:
        # Clean input if it's a dictionary string
        if isinstance(company_name, str) and company_name.startswith("{"):
            try:
                data = json.loads(company_name)
                company_name = data.get("symbol", "")
            except:
                pass
                
        logger.info(f"Fetching stock data for {company_name}")
        
        # Get the symbol
        symbol = get_symbol_for_company(company_name)
        logger.info(f"Using symbol {symbol}")

        # Try to get data with the symbol
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info:
            # If no info found, try with .NS suffix
            if not symbol.endswith(('.NS', '.BO')):
                logger.info(f"No data found for {symbol}, trying with .NS suffix")
                symbol = f"{symbol}.NS"
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # If still no info, try with .BO suffix
                if not info:
                    logger.info(f"No data found for {symbol}, trying with .BO suffix")
                    symbol = symbol[:-3] + ".BO"  # Replace .NS with .BO
                    stock = yf.Ticker(symbol)
                    info = stock.info
        
        if not info:
            logger.error(f"No data found for symbol: {symbol}")
            return {"error": f"Could not fetch data for {company_name} ({symbol})"}

        # Get weekly data for the last 5 years
        weekly_data = stock.history(period="5y", interval="1wk")
        
        # Get the last 5 weeks of data
        last_5_weeks = format_stock_data(weekly_data.head(5))
        
        # Get all weekly data
        all_weekly_data = format_stock_data(weekly_data)
        
        # Add some basic company info
        company_info = {
            "name": info.get("longName", "N/A"),
            "current_price": str(info.get("currentPrice", "N/A")),
            "market_cap": str(info.get("marketCap", "N/A")),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A")
        }
        
        return {
            "company_info": company_info,
            "last_5_weeks": last_5_weeks,
            "all_weekly_data": all_weekly_data
        }
        
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        return {"error": f"An error occurred while fetching stock data: {str(e)}"}


