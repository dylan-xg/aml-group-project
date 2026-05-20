
import cv2 as cv

from src.panopticon.settings import SETTINGS
from src.panopticon.typing import Frame
from src.panopticon.interface import UserInterface, VideoWidget
from src.panopticon.video_processor import VideoFeed


def demonstrate_callback(frame: Frame) -> Frame:
	return cv.flip(src=frame, flipCode=0)

if not SETTINGS.TESTING_VID:
	raise ValueError('`TESTING_VID` setting not set.')

if not SETTINGS.TESTING_VID.exists():
	raise ValueError(f'File not found: {SETTINGS.TESTING_VID}')

app = UserInterface(
	width=1000,
	height=500,
	title='Test Window'
)

video_feed = VideoFeed(
	capture_location=SETTINGS.TESTING_VID,
	callback=demonstrate_callback,
	frametime=1/60
)

app.add_feed(
	VideoWidget(
		parent_frame=app.video_container,
		video_feed=video_feed
	)
)

app.start()
