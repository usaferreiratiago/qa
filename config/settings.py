from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    
    # This configuration ignores any other keys found in your .env
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()