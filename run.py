
from pathlib import Path

import cv2 as cv

from src.panopticon.typing import Frame
from src.panopticon.interface import UserInterface, VideoWidget
from src.panopticon.video_processor import VideoFeed


# You will need to add your own video here
VID_PATH = Path('data/testing/example.mp4')

def demonstrate_callback(frame: Frame) -> Frame:
	return cv.flip(src=frame, flipCode=0)


if not VID_PATH.exists():
	print(f'File not found: {VID_PATH}')
	quit()

app = UserInterface(
	width=1000,
	height=500,
	title='Test Window'
)

video_feed = VideoFeed(
	capture_location=VID_PATH,
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
