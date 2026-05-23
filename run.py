"""Run Panopticon, our face recognition program."""

import cv2 as cv

from src.panopticon.recognition import fake_load_models
from src.panopticon.settings import SETTINGS
from src.panopticon.typing import Frame
from src.panopticon.ui import UserInterface
from src.panopticon.video_feed import VideoFeed
from src.panopticon.recognition.testing import fake_load_models


def demonstrate_callback(frame: Frame) -> Frame:
	return cv.flip(src=frame, flipCode=1)


video_feed = VideoFeed(
	capture_location=SETTINGS.input_source(),
	callback=demonstrate_callback,
	frametime=1./SETTINGS.FRAMERATE
)

app = UserInterface(
	title='Test Window',
	video_feed=video_feed
)

app.add_models(models=fake_load_models())

app.start()
