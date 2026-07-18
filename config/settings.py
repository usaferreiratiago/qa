from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # This maps the environment variable GEMINI_API_KEY to the python attribute GOOGLE_API_KEY
    GOOGLE_API_KEY: str = Field(..., validation_alias="GEMINI_API_KEY")

    DEBUG_MODE = True
    
    # Keep your model here, or add it to .env and define it similarly
    MODEL: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

settings = Settings()