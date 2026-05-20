"""Project settings file.

Create a file named .env and set the variables to the values you want.

e.g.:
DEEPFACE_POSTGRES_URI='postgresql://postgres:@localhost/deepface'
"""

from pathlib import Path as _Path
from typing import Annotated as _Annotated

from pydantic import Field as _Field
from pydantic_settings import (
	BaseSettings as _BaseSettings,
	SettingsConfigDict as _SettingsConfigDict
)


class _Settings(_BaseSettings):
	model_config = _SettingsConfigDict(
		env_file='.env',
		env_file_encoding='utf-8',
		case_sensitive=True,
		env_ignore_empty=True,
		extra='ignore'
	)

	DEEPFACE_POSTGRES_URI: _Annotated[
		str | None,
		_Field(frozen=True)
	] = None
	"""This value isn't used directly, but it is needed for Deepface to work.

	Example
		`DEEPFACE_POSTGRES_URI='postgresql://postgres:@localhost/deepface'`
	"""

	TESTING_VID: _Annotated[
		_Path | None,
		_Field(frozen=True)
	] = None
	"""The path to a video using for testing.

	Example
		`TESTING_VID='data/testing/example.mp4'`
	"""

	WINDOW_WIDTH: _Annotated[
		int,
		_Field(frozen=True)
	] = 1000
	"""The starting width of the window."""

	WINDOW_HEIGHT: _Annotated[
		int,
		_Field(frozen=True)
	] = 500
	"""The starting height of the window."""

# Module-level singleton pattern
SETTINGS = _Settings()

#if not SETTINGS.DEEPFACE_POSTGRES_URI: raise ValueError('Postgres URI not set!')
