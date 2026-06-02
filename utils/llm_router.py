import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("llm_router")

# Import provider functions
try:
    from utils.provider_groq import generate_with_groq
    from utils.provider_openrouter import generate_with_openrouter
    from utils.provider_cohere import generate_with_cohere
except ImportError:
    from provider_groq import generate_with_groq
    from provider_openrouter import generate_with_openrouter
    from provider_cohere import generate_with_cohere

# Sequential list of providers for routing and fallback support
# Easy to extend with new providers in the future
PROVIDERS = [
    {
        "name": "Groq",
        "func": generate_with_groq
    },
    {
        "name": "OpenRouter",
        "func": generate_with_openrouter
    },
    {
        "name": "Cohere",
        "func": generate_with_cohere
    }
]

def route_prompt(prompt: str) -> Dict[str, Any]:
    """
    Accepts a prompt and tries LLM providers sequentially:
    1. Groq
    2. OpenRouter
    3. Cohere
    
    If a provider fails, logs the failure and automatically falls back to the next provider.
    Returns the first successful response standardized as:
    {
        "success": True/False,
        "provider": "ProviderName" or None,
        "response": "ResponseText" or None,
        "error": None or "Error details"
    }
    """
    attempts_errors = []
    
    for provider_info in PROVIDERS:
        provider_name = provider_info["name"]
        generate_func = provider_info["func"]
        
        logger.info(f"Attempting LLM generation with provider: {provider_name}")
        try:
            response_text = generate_func(prompt)
            logger.info(f"Successfully generated response using provider: {provider_name}")
            return {
                "success": True,
                "provider": provider_name,
                "response": response_text,
                "error": None
            }
        except Exception as e:
            error_message = f"Error in provider {provider_name}: {str(e)}"
            logger.warning(error_message)
            attempts_errors.append(error_message)
            # Automatic fallback to the next provider
            continue

    # If all providers failed
    combined_errors = " | ".join(attempts_errors)
    logger.error(f"All LLM providers failed to generate a response. Errors: {combined_errors}")
    return {
        "success": False,
        "provider": None,
        "response": None,
        "error": combined_errors
    }

if __name__ == "__main__":
    print("Testing routing orchestrator...")
    test_prompt = "Say hello in 3 words."
    res = route_prompt(test_prompt)
    print("\nOrchestrated Result:")
    print(res)
