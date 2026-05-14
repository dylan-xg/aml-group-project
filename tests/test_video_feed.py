"""A file for testing the video_feed module."""

from pathlib import Path

import pytest

from src.panopticon.video_processor import DisplayConfig, VideoFeed


VIDEO_PATH = Path('data/videos/example.mp4')
NEVER_PATH = Path('a/a/a/a/a/a/aa/a/a/aa/a/a/a/a/a/a/a/a/a/a/a')

def test_example_video():
	VideoFeed.process_video(capture_location=VIDEO_PATH)

def test_incorrect_path():
	with pytest.raises(FileNotFoundError):
		VideoFeed.process_video(capture_location=NEVER_PATH)

def test_invalid_webcam():
	with pytest.raises(ValueError):
		VideoFeed.process_video(capture_location=-1)
