import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search_tool = TavilySearchResults(max_results=5)
tavily_search_tool.description = "You can answer any questions related to finance, stock market, news, terms and conditions, loan, etc. and no other questions unrelated to finance."

