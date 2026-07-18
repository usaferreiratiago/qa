from pydantic import BaseSettings

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    DEPLOYMENT_NAME: str
    
    class Config:
        env_file = ".env"

settings = Settings()