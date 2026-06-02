import os
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_with_cohere(prompt: str) -> str:
    """
    Generate a response using Cohere's chat model command-a-03-2025.
    """
    api_key = os.getenv("cohere_key")
    if not api_key:
        raise ValueError("Cohere API key ('cohere_key') is not set in environment variables.")
        
    co = cohere.Client(api_key)
    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )
    
    if not response.text:
        raise RuntimeError("Cohere API returned an empty response.")
        
    return response.text

if __name__ == "__main__":
    try:
        print("Testing Cohere Provider...")
        test_prompt = "Summarize today's AI industry trends in 3 bullet points."
        result = generate_with_cohere(test_prompt)
        print("\nCOHERE RESPONSE:\n")
        print(result)
    except Exception as e:
        print(f"Error occurred: {e}")