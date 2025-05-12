import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Optional, Type, Any, ClassVar
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain.tools import BaseTool

# Simple HTTPException for error handling
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure keys are present
if not all([PINECONE_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY]):
    raise RuntimeError("Missing one or more required API keys: PINECONE_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
index_name = "financial-products"
try:
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,  # matches all-MiniLM-L12-v2 embedding dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
except Exception as e:
    print(f"Error creating Pinecone index: {e}")

# Initialize embeddings and vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
try:
    docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding,
        pinecone_api_key=PINECONE_API_KEY
    )
except Exception as e:
    print(f"Error initializing PineconeVectorStore: {e}")
    docsearch = None

# External web search tool
search = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)

# LLM for generative responses
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
    max_output_tokens=512
)

# Prompt template for financial advice
prompt_template = PromptTemplate(
    template="""
You are a seasoned financial advisor. Use the provided Context (which may include details on debit cards, credit cards, savings and current accounts from various banks) to answer the user's financial question. Provide clear, concise, and well-structured advice, including comparisons when relevant.

Context:
{context}

Question:
{question}

Answer (structured, actionable):""",
    input_variables=["context", "question"]
)

async def _retrieve_and_answer(question: str, top_k: int = 5) -> str:
    # 1. Retrieve similar docs
    try:
        similar_docs = docsearch.similarity_search(question, k=top_k)
        context = "\n".join([doc.page_content for doc in similar_docs])
    except Exception:
        context = ""

    # 2. External web lookup (e.g., latest fees)
    try:
        web_results = search.invoke(question)
        extra = [res.get('content', '') for res in web_results]
        context += "\n" + "\n".join(extra)
    except Exception:
        pass

    # 3. Generate prompt and invoke LLM
    prompt = prompt_template.format(context=context, question=question)
    resp = llm.invoke(prompt)
    return getattr(resp, 'content', str(resp))

class FinancialAdvisorInput(BaseModel):
    query: str = Field(description="The financial question or query to get advice on")

class FinancialAdvisorTool(BaseTool):
    name: str = "financial_advisor"
    description: str = "A tool that provides general financial advice and answers questions about personal finance, investments, and money management."
    args_schema: Type[BaseModel] = FinancialAdvisorInput

    def _run(self, query: str) -> str:
        # Simple financial advice logic
        query = query.lower()
        
        if "portfolio" in query or "rate my portfolio" in query:
            return """To properly analyze your portfolio, I'll need some information:
1. What types of investments do you currently hold? (stocks, bonds, mutual funds, etc.)
2. What is your current asset allocation?
3. What is your investment timeline?
4. What is your risk tolerance?
5. What are your financial goals?

Please provide these details, and I can give you a comprehensive portfolio analysis and recommendations for improvement."""
        
        elif "investment" in query or "invest" in query:
            return """Here are some general investment tips:
1. Diversify your portfolio across different asset classes
2. Consider your risk tolerance and investment timeline
3. Start with index funds or ETFs for beginners
4. Regularly review and rebalance your portfolio
5. Consider tax implications of your investments
6. Don't invest money you'll need in the short term
7. Consider consulting a financial advisor for personalized advice"""
        
        elif "saving" in query or "save money" in query:
            return """Here are some money-saving strategies:
1. Create and stick to a budget
2. Build an emergency fund (3-6 months of expenses)
3. Pay off high-interest debt first
4. Automate your savings
5. Look for ways to reduce recurring expenses
6. Consider high-yield savings accounts
7. Take advantage of employer retirement matching"""
        
        elif "retirement" in query:
            return """Key retirement planning tips:
1. Start saving early and consistently
2. Take advantage of tax-advantaged accounts (401(k), IRA)
3. Consider your retirement lifestyle and expenses
4. Plan for healthcare costs
5. Diversify your retirement savings
6. Consider working with a retirement planner
7. Regularly review and adjust your retirement plan"""
        
        elif "debt" in query:
            return """Debt management strategies:
1. List all your debts and interest rates
2. Consider the debt snowball or avalanche method
3. Look for ways to lower interest rates
4. Create a debt payoff plan
5. Avoid taking on new debt
6. Consider debt consolidation if appropriate
7. Build an emergency fund to prevent new debt"""
        
        else:
            return """I can help you with various financial topics including:
- Portfolio analysis and management
- Investment strategies
- Saving money
- Retirement planning
- Debt management
- Budgeting
- Financial goal setting

Please ask a specific question about any of these topics, and I'll provide detailed advice."""

    async def _arun(self, query: str) -> str:
        return self._run(query)

financial_advisor = FinancialAdvisorTool()

# Example usage
if __name__ == "__main__":
    q = "Compare the cashback benefits of HDFC vs SBI credit cards for a monthly spend of ₹50,000"
    result = asyncio.run(_retrieve_and_answer(q))
    print("Advice:", result)
