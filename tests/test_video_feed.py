"""A file for testing the video_feed module."""

from pathlib import Path

import pytest

from src.panopticon import video_processor as VP
from src.panopticon.settings import SETTINGS


def test_example_video():
	if not SETTINGS.TESTING_VID:
		raise ValueError('`TESTING_VID` setting not set.')

	if not SETTINGS.TESTING_VID.exists():
		raise ValueError(f'File not found: {SETTINGS.TESTING_VID}')

	VP.process_video(capture_location=SETTINGS.TESTING_VID)

def test_incorrect_path():
	# Construct a path that does not exist
	never_path = Path('a/')
	while (never_path.exists()):
		never_path = never_path / 'a/'

	with pytest.raises(FileNotFoundError):
		VP.process_video(capture_location=never_path)

def test_invalid_webcam():
	with pytest.raises(ValueError):
		VP.process_video(capture_location=-1)
