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
	SettingsConfigDict as _SettingsConfigDict,
)


INPUT_TYPE_WEBCAM = "webcam"
INPUT_TYPE_VIDEO = "video"


class _Settings(_BaseSettings):
	model_config = _SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		env_ignore_empty=True,
		extra="ignore",
	)

	WINDOW_WIDTH: _Annotated[int, _Field(frozen=True, gt=1)] = 1000
	"""The starting width of the window."""

	WINDOW_HEIGHT: _Annotated[int, _Field(frozen=True, gt=1)] = 500
	"""The starting height of the window."""

	INPUT_SOURCE: _Annotated[str, _Field(frozen=True)] = "webcam"
	"""What input source to use.

	options: 'webcam', 'video'
	"""

	WEBCAM_ID: _Annotated[int, _Field(frozen=True, ge=0)] = 0
	"""The id of the webcam used."""

	TESTING_VID: _Annotated[_Path | None, _Field(frozen=True)] = None
	"""The path to a video using for testing.

	Example
		`TESTING_VID='data/testing/example.mp4'`
	"""

	FRAMERATE: _Annotated[float, _Field(frozen=True, gt=0)] = 30.0
	"""The maximum framerate that is displayed.

	The actual framerate will likely be lower due to peformance overhead.
	"""

	INPUT_BUTTON_WIDTH: _Annotated[int, _Field(frozen=True, gt=0, lt=WINDOW_WIDTH)] = 30
	"""The width to use for the input button."""

	MODEL_WEIGHTS_LOCATION: _Annotated[_Path, _Field(frozen=True)] = _Path(
		"src/panopticon/model_weights"
	)
	"""The path to the folder containing the model weights."""

	USE_POSTGRES_DB: _Annotated[bool, _Field(frozen=True)] = False
	"""Whether to use the stateless database search."""

	DEEPFACE_POSTGRES_URI: _Annotated[str | None, _Field(frozen=True)] = None
	"""If :func:`USE_POSTGRES_DB` is True, what is the URI.

	This value isn't used directly, but it is needed for Deepface to work.

	Example
		`DEEPFACE_POSTGRES_URI='postgresql://postgres:@localhost/deepface'`
	"""

	LOCAL_DATABASE_PATH: _Annotated[_Path, _Field(frozen=True)] = _Path(
		"data/faces_db"
	)
	"""If :func:`USE_POSTGRES_DB` is False, what folder is the database."""

	DETECTOR_BACKEND: _Annotated[str, _Field(frozen=True)] = "ssd"
	"""The detector backend that DeepFace will use."""

	DISTANCE_METRIC: _Annotated[str, _Field(frozen=True)] = "euclidean_l2"
	"""The distance metric that DeepFace will use."""

	TOP_K: _Annotated[int, _Field(frozen=True, gt=0)] = 1
	"""How many matches to return for each face."""

	MODEL_NAME: _Annotated[str, _Field(frozen=True)] = "face_classifier"
	"""The name used to identify our model in DeepFace."""

	def input_source(self) -> int | _Path:
		"""Util function to parse and validate the input source.

		Returns
		-------
		INPUT : int or Path
			If the input source is determined to be a:
			- webcam, return the integer ID of the webcam.
			- video, validate that the file exists. Does not confirm the file type.

		Raises
		------
		ValueError
			- If an incorrect input source was somehow set.
			- If an input video was selected but not set.
			- If an input video was selected but set to an invalid path.
		"""
		if self.INPUT_SOURCE == INPUT_TYPE_WEBCAM:
			INPUT = self.WEBCAM_ID
		elif self.INPUT_SOURCE == INPUT_TYPE_VIDEO:
			# Technically redundant
			if not self.TESTING_VID:
				raise ValueError("`TESTING_VID` setting not set.")
			if not self.TESTING_VID.exists():
				raise ValueError(f"File not found: {self.TESTING_VID}")

			INPUT = self.TESTING_VID
		else:
			raise ValueError(
				f"INPUT_SOURCE somehow set to invalid value. {self.INPUT_SOURCE}"
			)

		return INPUT


SETTINGS = _Settings()
"""All of the project settings.

Module level singleton pattern.
"""

# if not SETTINGS.DEEPFACE_POSTGRES_URI: raise ValueError('Postgres URI not set!')

# Verify the input video if it is chosen.
if SETTINGS.INPUT_SOURCE == INPUT_TYPE_VIDEO:
	if not SETTINGS.TESTING_VID:
		raise ValueError("`TESTING_VID` setting not set.")
	if not SETTINGS.TESTING_VID.exists():
		raise ValueError(f"File not found: {SETTINGS.TESTING_VID}")
