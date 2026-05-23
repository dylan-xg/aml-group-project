
import cv2 as cv

from src.panopticon.recognition import fake_load_models
from src.panopticon.settings import SETTINGS
from src.panopticon.typing import Frame
from src.panopticon.ui import UserInterface
from src.panopticon.video_feed import VideoFeed


def demonstrate_callback(frame: Frame) -> Frame:
	return cv.flip(src=frame, flipCode=1)


video_feed = VideoFeed(
	capture_location=SETTINGS.input_source(),
	callback=demonstrate_callback,
	frametime=1/60
)

#webcam_feed = VideoFeed(capture_location=0)

app = UserInterface(
	title='Test Window',
	video_feed=video_feed
)

#def load_video_feed(ui: UserInterface, /) -> None:
#	ui.add_feed(video_feed)

#def load_webcam_feed(ui: UserInterface, /) -> None:
#	ui.add_feed(webcam_feed)

#app.add_button(
#	label='Load test video',
#	order=0,
#	command=lambda: load_video_feed(app)
#)

#app.add_button(
#	label='Load webcam',
#	order=1,
#	command=lambda: load_webcam_feed(app)
#)

#app.add_feed(video_feed)

app.add_button(label='Do nothing', order=1000, command=lambda:None)

app.add_models(models=fake_load_models())

app.start()
