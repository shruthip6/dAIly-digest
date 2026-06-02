import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_with_groq(prompt: str) -> str:
    """
    Generate a response using Groq's model llama-3.3-70b-versatile.
    """
    api_key = os.getenv("grok_key")
    if not api_key:
        raise ValueError("Groq API key ('grok_key') is not set in environment variables.")
        
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    
    if not chat_completion.choices or not chat_completion.choices[0].message.content:
        raise RuntimeError("Groq API returned an empty response.")
        
    return chat_completion.choices[0].message.content

# if __name__ == "__main__":
#     try:
#         print("Testing Groq Provider...")
#         test_prompt = "Summarize today's AI industry trends in 3 bullet points."
#         result = generate_with_groq(test_prompt)
#         print("\nGROQ RESPONSE:\n")
#         print(result)
#     except Exception as e:
#         print(f"Error occurred: {e}")