import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_with_openrouter(prompt: str) -> str:
    """
    Generate a response using OpenRouter's model deepseek/deepseek-chat.
    """
    api_key = os.getenv("open_router_key")
    if not api_key:
        raise ValueError("OpenRouter API key ('open_router_key') is not set in environment variables.")
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    if not completion.choices or not completion.choices[0].message.content:
        raise RuntimeError("OpenRouter API returned an empty response.")
        
    return completion.choices[0].message.content

# if __name__ == "__main__":
#     try:
#         print("Testing OpenRouter Provider...")
#         test_prompt = "Summarize today's AI industry trends in 3 bullet points."
#         result = generate_with_openrouter(test_prompt)
#         print("\nOPENROUTER RESPONSE:\n")
#         print(result)
#     except Exception as e:
#         print(f"Error occurred: {e}")