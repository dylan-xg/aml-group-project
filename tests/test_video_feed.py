"""A file for testing the video_feed module."""

from pathlib import Path

import pytest

from src.panopticon import video_processor as VP


VIDEO_PATH = Path('data/testing/example.mp4')
NEVER_PATH = Path('a/a/a/a/a/a/aa/a/a/aa/a/a/a/a/a/a/a/a/a/a/a')

def test_example_video():
	VP.process_video(capture_location=VIDEO_PATH)

def test_incorrect_path():
	with pytest.raises(FileNotFoundError):
		VP.process_video(capture_location=NEVER_PATH)

def test_invalid_webcam():
	with pytest.raises(ValueError):
		VP.process_video(capture_location=-1)
