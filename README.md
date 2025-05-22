# Agentify Financial Assistant

An AI-powered financial assistant platform that provides comprehensive financial analysis, portfolio management, loan assessment, and investment advice using advanced language models and specialized tools.

## Created By

- [Kabir Chawla](https://github.com/kabirchawla2003)
- [Harshit Aggarwal](https://github.com/Blasterharsh99)
- [Harsh Pandey](https://github.com/Harshpandey22)

## Features

- **Stock Market Analysis**: Real-time stock data analysis and trend tracking
- **Portfolio Management**: Integration with Zerodha for portfolio analysis and optimization
- **Loan Assessment**: Advanced loan eligibility prediction and recommendations
- **Investment Advisory**: Personalized financial advice and investment strategies
- **Document Analysis**: Financial document and terms & conditions analysis
- **Market Research**: Real-time market research and news analysis

## Prerequisites

- Python 3.9+
- Environment variables configured in `.env` file
  - `GOOGLE_API_KEY`: Google AI API key
  - Other API keys as required by individual tools

## Installation

1. Clone the repository:
```sh
git clone https://github.com/yourusername/Agentify.git
cd Agentify
```

2. Install required packages:
```sh
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Running the API Server

Start the FastAPI server:
```sh
python agent_api.py
```

The server will start at `http://localhost:8000`

### Running the Web Interface

1. Navigate to the website directory:
```sh
cd website
```

2. Install dependencies:
```sh
pnpm install
```

3. Start the development server:
```sh
pnpm dev
```

The web interface will be available at `http://localhost:3000`

## API Endpoints

- `POST /query`: Process financial queries
  ```json
  {
    "query": "Your financial question here"
  }
  ```
- `GET /health`: Health check endpoint
- `GET /`: Serves the web interface

## Project Structure

```
├── agent_api.py            # Main API server
├── bank_advisor_tool.py    # Financial advice tool
├── loan_approver_tool.py   # Loan assessment tool
├── portfolio_zerodha.py    # Portfolio management tool
├── stock_analyzer_tool.py  # Stock analysis tool
├── tc_tool.py             # Terms analysis tool
├── web_search_tool.py     # Market research tool
└── website/               # Next.js web interface
    ├── app/              # Next.js app directory
    ├── components/       # React components
    ├── lib/             # Utility functions
    ├── styles/          # Global styles
    └── public/          # Static assets
```

## Tech Stack

- **Backend**:
  - FastAPI
  - LangChain
  - Google Generative AI
  - Various Python libraries (pandas, scikit-learn, etc.)

- **Frontend**:
  - Next.js
  - Tailwind CSS
  - React
  - shadcn/ui components

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Google Generative AI](https://ai.google.dev/)
- [shadcn/ui](https://ui.shadcn.com/)
