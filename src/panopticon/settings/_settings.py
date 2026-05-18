"""Project settings file.

Create a file named .env and set the variables to the values you want.

e.g.:
DEEPFACE_POSTGRES_URI='postgresql://postgres:@localhost/deepface'
"""

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
	# You need to set this in your .env file, but it doesn't need to be loaded here.
	#DEEPFACE_POSTGRES_URI: str = Field(default='postgresql://postgres:@localhost/deepface')

	# To shut up the linter
	null: bool = Field(default=False)

# Module-level singleton pattern
SETTINGS = Settings()
