from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # These fields will be read automatically from your .env file
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    DEPLOYMENT_NAME: str
    
    # Modern Pydantic V2 syntax for configuration
    model_config = SettingsConfigDict(env_file=".env")

# Initialize the settings object
settings = Settings()