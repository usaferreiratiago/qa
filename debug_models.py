from google import genai
from config.settings import settings

# Initialize client with the API key from your settings
client = genai.Client(api_key=settings.GOOGLE_API_KEY)

# List available models
models = client.models.list()

print("Available models:")
for model in models:
    print(f"Name: {model.name}, Display Name: {model.display_name}")