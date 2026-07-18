# debug_models.py
from google import genai
from config.settings import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)
models = client.models.list()

print("Available models:")
for model in models:
    print(f"Name: {model.name}, Display Name: {model.display_name}")