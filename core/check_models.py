import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

print("Available models that support generateContent:\n")
for model in client.models.list():
    if hasattr(model, 'supported_actions') and 'generateContent' in (model.supported_actions or []):
        print(f"  {model.name}")
    else:
        # fallback: print all models
        print(f"  {model.name}")
