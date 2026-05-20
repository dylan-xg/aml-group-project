"""A file for testing the video_feed module."""

from pathlib import Path

import pytest

from src.panopticon import video_processor as VP
from src.panopticon.settings import SETTINGS


NEVER_PATH = Path('a/a/a/a/a/a/aa/a/a/aa/a/a/a/a/a/a/a/a/a/a/a')

def test_example_video():
	if not SETTINGS.TESTING_VID:
		raise ValueError('`TESTING_VID` setting not set.')

	if not SETTINGS.TESTING_VID.exists():
		raise ValueError(f'File not found: {SETTINGS.TESTING_VID}')

	VP.process_video(capture_location=SETTINGS.TESTING_VID)

def test_incorrect_path():
	with pytest.raises(FileNotFoundError):
		VP.process_video(capture_location=NEVER_PATH)

def test_invalid_webcam():
	with pytest.raises(ValueError):
		VP.process_video(capture_location=-1)
