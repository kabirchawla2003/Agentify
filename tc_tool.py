import os
from dotenv import load_dotenv
import re
import asyncio # <-- Import asyncio
from typing import Dict, List
from langchain_core.tools import tool

# Pinecone client and spec
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# LangChain components for embeddings and LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Tavily search tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Prompting
from langchain.prompts import PromptTemplate

# Assuming HTTPException might be used in a web framework context later
# If running standalone, standard exceptions are fine. For simplicity here,
# we'll keep it but note it won't behave like in FastAPI/Starlette directly.
# from fastapi import HTTPException (or similar - remove if not needed)
# Define a placeholder if not using a web framework:
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


# 1. Load environment variables
load_dotenv()
PINECONE_API_KEY = "pcsk_DKaud_QY3WnSd6UrypA5nPuNMGxr3wgVSZHZCcZirrHaqcVJU1TmK5MrHYoBa6HXLu5nX"
# Use a default region if not set, or ensure it's set in .env
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1") # Example default
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Basic Checks for API Keys ---
if not all([PINECONE_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY]):
    print("Warning: One or more API keys (PINECONE_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY) are missing.")
    print("Please set them in your environment variables or a .env file.")
    # Decide if you want to exit or proceed with potential errors
    # exit(1) # Uncomment to exit if keys are crucial

# 2. Initialize Pinecone client and index
# Ensure PINECONE_ENV is a valid region for your chosen cloud
# Pinecone serverless regions: https://docs.pinecone.io/docs/regions#serverless
# e.g., "aws": ["us-east-1", "us-west-2"], "gcp": ["us-central1", "us-east1"]
# Check if PINECONE_ENV is compatible with the cloud='aws' below.
# Example: If PINECONE_ENV="us-west1-gcp", change cloud to "gcp"
pinecone_cloud = "aws"
pinecone_region = PINECONE_ENV

print(f"Using Pinecone Cloud: {pinecone_cloud}, Region: {pinecone_region}") # Debug print

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "analyze-terms"

existing_index_names = [] # Default to empty list
if PINECONE_API_KEY: # Only attempt Pinecone operations if key exists
    try:
        print("Fetching list of existing Pinecone indexes...")
        index_list_obj = pc.list_indexes()  # Returns the IndexList object
        existing_index_names = [idx_info['name'] for idx_info in index_list_obj]
        # --- END CORRECTION ---

        print(f"Found index names: {existing_index_names}")  # Should now print the list of names
    except AttributeError:
        # Fallback if the structure isn't directly iterable dicts with 'name'
        print("Warning: Could not directly extract names. Trying index_list_obj.names() if it's callable.")
        try:
            # If .names really *is* a method, try calling it? (Unusual)
            existing_index_names = index_list_obj.names()
            if not isinstance(existing_index_names, list): existing_index_names = []
        except:
            print("Error: Failed to get index names using iteration or calling .names().")
            existing_index_names = []  # Default to empty list on error
    except Exception as e:
        print(f"Error processing Pinecone index list: {e}")
        existing_index_names = []  # Ensure it's a list on error

    # Now check if the desired index_name is in the fetched list
    if index_name not in existing_index_names:
        print(f"Creating Pinecone index '{index_name}'...")
        try:
            pc.create_index(
                name=index_name,
                dimension=384, # Matches all-MiniLM-L12-v2
                metric="cosine",
                spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)
            )
            print(f"Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            # Decide how to handle this (e.g., exit or try to continue without Pinecone)
    else:
        print(f"Pinecone index '{index_name}' already exists.")
else:
    print("Skipping Pinecone index check due to missing API key.")

# 3. Build embeddings
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L12-v2'
)

# --- Continue with initializing PineconeVectorStore ---
# Ensure index exists before initializing, or handle potential errors
# Check against the potentially updated list of names
if PINECONE_API_KEY and index_name in existing_index_names: # Check against the variable
     docsearch = PineconeVectorStore(
         # pinecone_api_key=PINECONE_API_KEY, # Not needed if Pinecone client 'pc' is initialized
         index_name=index_name,
         embedding=embedding
         # environment=PINECONE_ENV, # Deprecated for newer client versions
     )
else:
     # Handle case where index still doesn't exist (e.g., creation failed) or API key missing
     if PINECONE_API_KEY and index_name not in existing_index_names:
         print(f"Warning: Index '{index_name}' not found or creation failed. PineconeVectorStore not initialized.")
     elif not PINECONE_API_KEY:
         print("Warning: PineconeVectorStore not initialized due to missing API key.")
     docsearch = None # Set to None so later code can check




# 4. Wrap in LangChain PineconeVectorStore
# Ensure index exists before initializing, or handle potential errors
docsearch = PineconeVectorStore(
     # pinecone_api_key=PINECONE_API_KEY, # Not needed if Pinecone client 'pc' is initialized
     index_name=index_name,
     embedding=embedding)


# 5. Initialize external search and LLM
search = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None # Use gemini-flash if 2.0 isn't available or needed

if not search:
    print("Warning: Tavily search not initialized due to missing API key.")
if not llm:
    print("Warning: Google Generative AI LLM not initialized due to missing API key.")


# 6. Define categories of important terms
IMPORTANT_CATEGORIES = {
    "liability": [
        "limitation of liability",
        "disclaimer",
        "waiver",
        "indemnification",
        "hold harmless"
    ],
    "privacy": [
        "data collection",
        "data sharing",
        "personal information",
        "tracking",
        "cookies"
    ],
    "fees": [
        "charges",
        "payment",
        "subscription",
        "refund",
        "cancellation fee"
    ],
    "unusual_terms": [
        "arbitration",
        "class action",
        "jurisdiction",
        "governing law",
        "unilateral changes"
    ]
}

prompt_template = PromptTemplate(
    template="""
    You are a legal expert specializing in analyzing Terms and Conditions documents, particularly focusing on identifying potential hidden costs and user implications. Your task is to analyze the provided text and return the analysis in a structured JSON format.

    IMPORTANT: Your response MUST be a valid JSON object with the following structure:
    {{
        "summary_overview": "A brief 2-3 sentence summary of the key points",
        "key_clauses": {{
            "liability": ["List of important liability clauses with brief explanations"],
            "privacy": ["List of important privacy/data handling clauses"],
            "fees": ["List of important fee-related clauses"],
            "unusual_terms": ["List of any unusual or concerning terms"]
        }},
        "user_rights": ["List of key user rights and permissions"],
        "user_obligations": ["List of key user responsibilities and requirements"],
        "hidden_costs": ["List of potential hidden or unexpected costs"],
        "privacy_concerns": ["List of important privacy considerations"],
        "risk_assessment": {{
            "high_risk_areas": ["List of high-risk clauses or terms"],
            "recommendations": ["List of recommendations for users"]
        }}
    }}

    Example JSON response:
    {{
        "summary_overview": "This service agreement outlines a subscription-based service with automatic renewal, strict liability limitations, and mandatory arbitration. Users must agree to data collection and sharing practices.",
        "key_clauses": {{
            "liability": [
                "Limited liability to last 6 months of payments",
                "No liability for data breaches or service interruptions"
            ],
            "privacy": [
                "Data collection and sharing with third parties",
                "Use of cookies and tracking technologies"
            ],
            "fees": [
                "Automatic subscription renewal",
                "No refunds for partial periods",
                "Early termination fees apply"
            ],
            "unusual_terms": [
                "Mandatory arbitration clause",
                "Class action waiver",
                "Unilateral terms modification"
            ]
        }},
        "user_rights": [
            "Right to cancel subscription",
            "Right to access personal data"
        ],
        "user_obligations": [
            "Must provide accurate information",
            "Responsible for maintaining account security"
        ],
        "hidden_costs": [
            "Automatic renewal charges",
            "Early termination fees",
            "Third-party service fees"
        ],
        "privacy_concerns": [
            "Data sharing with third parties",
            "Extensive tracking and cookies",
            "Limited data protection guarantees"
        ],
        "risk_assessment": {{
            "high_risk_areas": [
                "Limited liability protections",
                "Mandatory arbitration",
                "Automatic renewals"
            ],
            "recommendations": [
                "Review privacy settings regularly",
                "Set calendar reminders for subscription renewals",
                "Consider data sharing implications"
            ]
        }}
    }}

    Context from similar documents and web search (if available):
    {context}

    Document to Analyze:
    {terms_text}

    Remember to:
    1. Keep the analysis concise and focused on the most important terms
    2. Ensure the response is valid JSON
    3. Include all required fields in the structure
    4. Use clear, concise language in the explanations
    """,
    input_variables=["context", "terms_text"]
)


def preprocess_terms(terms_text: str) -> str:
    """
    Preprocess the terms and conditions text: basic cleaning and sectioning.
    """
    # Basic cleaning: replace multiple whitespace with single space, trim ends
    cleaned_text = re.sub(r'\s+', ' ', terms_text).strip()

    # Attempt to split into sections based on common patterns (numbered lists, caps headings)
    # This is heuristic and might need refinement based on common T&C structures
    # Pattern: Number+dot, UppercaseLetter+dot, word SECTION, or ALL CAPS lines (min 5 chars)
    sections = re.split(r'(\n\s*\d+\.\s+|\n\s*[A-Z]\.\s+|\n\s*SECTION\s+\d+[:\s]|\n\s*[A-Z\s]{5,}:\s*\n)', cleaned_text, flags=re.IGNORECASE)

    formatted_text = cleaned_text # Start with cleaned text in case split fails

    # Reconstruct if sections were found, trying to preserve headings
    if len(sections) > 1:
        formatted_text = sections[0].strip() # Add text before the first delimiter
        for i in range(1, len(sections), 2):
            heading = sections[i].strip()
            content = sections[i + 1].strip() if i + 1 < len(sections) else ''
            # Add newline before heading for better readability if not already there
            if not formatted_text.endswith('\n'):
                 formatted_text += '\n'
            formatted_text += f"\n{heading}\n{content}"

    # Fallback: simple paragraph breaks if no structure found
    if formatted_text == cleaned_text:
        formatted_text = cleaned_text.replace('. ', '.\n') # Simple sentence splitting as fallback

    return formatted_text.strip()


def identify_important_terms(terms_text: str) -> Dict[str, List[str]]:
    """
    Identify sentences containing important keywords and categorize them.
    """
    found_terms = {cat: [] for cat in IMPORTANT_CATEGORIES}
    lower_text = terms_text.lower()
    # Split into sentences for better context capture
    sentences = re.split(r'(?<=[.!?])\s+', lower_text) # Split by sentences

    for category, keywords in IMPORTANT_CATEGORIES.items():
        for keyword in keywords:
            # Use word boundaries to avoid partial matches (e.g., 'charge' in 'rechargeable')
            keyword_pattern = r'\b' + re.escape(keyword) + r'\b'
            for sentence in sentences:
                if re.search(keyword_pattern, sentence):
                    # Add the whole sentence containing the keyword
                    found_terms[category].append(sentence.strip().capitalize())
                    # Optional: break after first match per sentence to avoid duplicates if multiple keywords in one sentence
                    # break
        # Remove duplicate sentences within a category
        found_terms[category] = list(dict.fromkeys(found_terms[category]))
    return found_terms


def analyze_terms_and_conditions(terms_text: str) -> dict:
    """Main function to analyze T&C"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service not available. Check GOOGLE_API_KEY.")

    try:
        print("Preprocessing text...")
        processed_text = preprocess_terms(terms_text)
        print("Identifying important term categories...")
        important_terms = identify_important_terms(processed_text)

        context = "No similar documents found in local index."
        # Retrieve similar docs from Pinecone if available
        if docsearch:
             try:
                 print("Searching Pinecone for similar documents...")
                 # Use a smaller chunk for similarity search if text is very long
                 search_query = processed_text[:1000]
                 similar_documents = docsearch.similarity_search(search_query, k=3)
                 if similar_documents:
                      context = "\n---\n".join([f"Similar Doc Snippet:\n{doc.page_content}" for doc in similar_documents])
                 else:
                      context = "No relevant documents found in Pinecone."
             except Exception as e:
                 print(f"Warning: Pinecone search failed: {e}")
                 context = "Error during Pinecone search."
        else:
             print("Skipping Pinecone search (not initialized).")


        internet_context = "Web search not performed or failed."
        # External web search if available
        if search:
            try:
                print("Performing Tavily web search...")
                # Use a concise query based on the start of the text
                web_query = "Key points and risks in terms and conditions like: " + terms_text[:150]
                internet_data = search.invoke(web_query) # Tavily invoke takes string directly
                # Process Tavily results (assuming it returns a list of dicts with 'content')
                if isinstance(internet_data, list) and internet_data:
                     internet_context = "\n---\n".join([f"Web Search Result:\n{item.get('content', 'No content found')}" for item in internet_data])
                else:
                     internet_context = "No relevant results from web search."

            except Exception as e:
                print(f"Warning: Tavily search failed: {e}")
                internet_context = "Error during web search."
        else:
             print("Skipping Tavily search (not initialized).")

        full_context = f"Context from Vector DB:\n{context}\n\nContext from Web Search:\n{internet_context}"

        # LLM analysis
        print("Invoking LLM for analysis...")
        prompt_input = prompt_template.format(context=full_context, terms_text=processed_text)

        # Ensure llm.invoke receives a string. If using older Langchain versions expecting messages, adjust here.
        if isinstance(prompt_input, str):
             analysis_result = llm.invoke(prompt_input)
             # Assuming the result has a 'content' attribute with the response string
             analysis_content = getattr(analysis_result, 'content', str(analysis_result))
        else:
             # Handle cases where invoke might expect a different format (e.g., list of messages)
             # This depends on the specific LangChain version and model wrapper
             # Example for message format:
             # from langchain_core.messages import HumanMessage
             # analysis_result = llm.invoke([HumanMessage(content=prompt_input)])
             # analysis_content = getattr(analysis_result, 'content', str(analysis_result))
             print("Warning: LLM input format might need adjustment for this Langchain version.")
             analysis_content = "Error: LLM invocation format issue."


        print("LLM analysis complete.")

        # Attempt to parse the JSON output from the LLM
        try:
            # Clean potential markdown ```json ... ``` markers
            cleaned_analysis_content = re.sub(r'^```json\s*|\s*```$', '', analysis_content, flags=re.MULTILINE).strip()
            # Load the cleaned string as JSON
            parsed_analysis = json.loads(cleaned_analysis_content)
            detailed_analysis_output = parsed_analysis # Store the parsed JSON object
        except json.JSONDecodeError as json_e:
            print(f"Warning: LLM output was not valid JSON. Error: {json_e}")
            print("Raw LLM output:", analysis_content) # Log the raw output for debugging
            # Fallback: return the raw string within a basic structure
            detailed_analysis_output = {
                "error": "LLM did not return valid JSON.",
                "raw_output": analysis_content
            }
        except Exception as e:
             print(f"Error processing LLM response: {e}")
             detailed_analysis_output = {
                "error": "Failed to process LLM response.",
                "raw_output": analysis_content
            }

        return {
            # Return the parsed JSON directly if successful, or the error structure
            "detailed_analysis": detailed_analysis_output,
            "important_terms_by_category": important_terms,
            "processed_text_preview": processed_text[:1000] + "..." # Return a preview
        }
    except HTTPException:
         # Re-raise HTTPExceptions if they occur (e.g., LLM not available)
         raise
    except Exception as e:
        print(f"Unexpected error in analyze_terms_and_conditions: {e}") # Log unexpected errors
        # Raise as a generic internal server error in a web context,
        # or just re-raise for script execution.
        raise HTTPException(status_code=500, detail=f"Error analyzing terms and conditions: {str(e)}")

@tool
def analyze_terms(terms_text: str) -> Dict:
    """
    Analyzes terms and conditions text, summarizes key points, and flags important clauses.

    Args:
        terms_text (str): The terms and conditions to analyze.

    Returns:
        important_terms (str): Summary and analysis of the terms.

    Raises:
        HTTPException: If the input is too short or if an error occurs during processing.
    """
    if not terms_text or len(terms_text.strip()) < 100:
        # In a standalone script, print error or raise standard Exception
        # raise ValueError("Please provide terms and conditions text of at least 100 characters.")
        # Using HTTPException for consistency if planned for web use
         raise HTTPException(status_code=400,
                             detail="Please provide terms and conditions text of at least 100 characters.")
    # No need to run in executor if analyze_terms_and_conditions is already synchronous
    # If it were async, you might use loop.run_in_executor
    try:
        result = analyze_terms_and_conditions(terms_text)
        return result
    except HTTPException as http_exc:
         # Log or handle specific HTTP errors if needed
         print(f"Analysis failed with status {http_exc.status_code}: {http_exc.detail}")
         # Re-raise to be caught by the caller (e.g., main block or web framework)
         raise
    except Exception as e:
         print(f"An unexpected error occurred during analysis: {e}")
         # Wrap unexpected errors in a standard HTTPException for consistency
         raise HTTPException(status_code=500, detail=f"Unexpected analysis error: {str(e)}")


# <-- Add more substantial placeholder text -->
s = """
SERVICE AGREEMENT AND TERMS OF USE

1. ACCEPTANCE OF TERMS
By accessing or using the services provided by ExampleCorp ('Service'), you agree to be bound by these Terms of Use ('Terms'). If you disagree with any part of the terms, then you may not access the Service. We may update these terms unilaterally without notice. Continued use constitutes acceptance.

2. DESCRIPTION OF SERVICE
ExampleCorp provides users with access to a rich collection of resources, including various communications tools, forums, personalized content and branded programming through its network of properties which may be accessed through any various medium or device now known or hereafter developed.

3. USER OBLIGATIONS
You are responsible for obtaining access to the Service, and that access may involve third-party fees (such as Internet service provider or airtime charges). You are responsible for those fees. In addition, you must provide and are responsible for all equipment necessary to access the Service. You agree not to access the Service by any means other than through the interface that is provided by ExampleCorp.

4. PRIVACY POLICY
Registration Data and certain other information about you are subject to our applicable privacy policy. For more information, see the full Privacy Policy. You understand that through your use of the Service you consent to the collection and use (as set forth in the applicable privacy policy) of this information, including the transfer of this information for storage, processing and use by ExampleCorp and its affiliates. We utilize cookies and tracking technologies. Data may be shared with third-party partners for advertising purposes.

5. FEES AND PAYMENT
Certain aspects of the Service may be provided for a fee or other charge. If you elect to use paid aspects of the Service, you agree to the pricing and payment terms, as we may update them from time to time. ExampleCorp may add new services for additional fees and charges, or amend fees and charges for existing services, at any time in its sole discretion. All subscriptions automatically renew unless cancelled 24 hours before the renewal date. No refunds are provided for partial subscription periods. A cancellation fee may apply if you terminate your contract early.

6. DISCLAIMER OF WARRANTIES
YOUR USE OF THE SERVICE IS AT YOUR SOLE RISK. THE SERVICE IS PROVIDED ON AN 'AS IS' AND 'AS AVAILABLE' BASIS. EXAMPLECORP EXPRESSLY DISCLAIMS ALL WARRANTIES OF ANY KIND, WHETHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.

7. LIMITATION OF LIABILITY
YOU EXPRESSLY UNDERSTAND AND AGREE THAT EXAMPLECORP AND ITS SUBSIDIARIES, AFFILIATES, OFFICERS, EMPLOYEES, AGENTS, PARTNERS AND LICENSORS SHALL NOT BE LIABLE TO YOU FOR ANY PUNITIVE, INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR EXEMPLARY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DAMAGES FOR LOSS OF PROFITS, GOODWILL, USE, DATA OR OTHER INTANGIBLE LOSSES (EVEN IF EXAMPLECORP HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES), RESULTING FROM: (a) THE USE OR THE INABILITY TO USE THE SERVICE; (b) THE COST OF PROCUREMENT OF SUBSTITUTE GOODS AND SERVICES; (c) UNAUTHORIZED ACCESS TO OR ALTERATION OF YOUR TRANSMISSIONS OR DATA; (d) STATEMENTS OR CONDUCT OF ANY THIRD PARTY ON THE SERVICE; OR (e) ANY OTHER MATTER RELATING TO THE SERVICE. OUR TOTAL LIABILITY IS LIMITED TO THE AMOUNT YOU PAID US IN THE LAST 6 MONTHS.

8. INDEMNIFICATION
You agree to indemnify and hold ExampleCorp and its subsidiaries, affiliates, officers, agents, employees, partners and licensors harmless from any claim or demand, including reasonable attorneys' fees, made by any third party due to or arising out of Content you submit, post, transmit, modify or otherwise make available through the Service, your use of the Service, your connection to the Service, your violation of the Terms, or your violation of any rights of another.

9. GOVERNING LAW AND JURISDICTION
These Terms shall be governed by and construed in accordance with the laws of the State of ExampleState, without regard to its conflict of law provisions. You agree to submit to the personal and exclusive jurisdiction of the courts located within the county of ExampleCounty, ExampleState.

10. ARBITRATION CLAUSE
Any dispute arising from these Terms will be resolved solely through final and binding arbitration, rather than in court. You waive any right to participate in a class action lawsuit or class-wide arbitration.
"""

# <-- Need to import json to parse the LLM output -->
import json

if __name__ == "__main__":
    # Check if required keys are present before running
    if not all([PINECONE_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY]):
        print("Execution cannot proceed without required API keys. Exiting.")
    else:
        print("Starting Terms and Conditions analysis...")
        try:
            # <-- Run the async function using asyncio.run() -->
            analysis_result = asyncio.run(analyze_terms(s))
            print("\n--- Analysis Result ---")
            # Pretty print the resulting dictionary
            import pprint
            pprint.pprint(analysis_result)
            print("--- End of Analysis ---")

            # Example: Accessing specific parts of the analysis if JSON was parsed correctly
            if isinstance(analysis_result.get("detailed_analysis"), dict) and "error" not in analysis_result.get("detailed_analysis"):
                print("\nAccessing specific fields (example):")
                summary = analysis_result["detailed_analysis"].get("summary_overview", "N/A")
                print(f"Summary Overview: {summary}")
                privacy = analysis_result["detailed_analysis"].get("privacy_summary", "N/A")
                print(f"Privacy Summary: {privacy}")
            elif isinstance(analysis_result.get("detailed_analysis"), dict) and "error" in analysis_result.get("detailed_analysis"):
                 print(f"\nLLM Analysis Error: {analysis_result['detailed_analysis']['error']}")


        except HTTPException as e:
             print(f"\nAnalysis failed with status code {e.status_code}: {e.detail}")
        except Exception as e:
             print(f"\nAn unexpected error occurred in the main block: {e}")