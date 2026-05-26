"""The entry point for panopticon."""

from .modules import LOADED_MODULES
from .settings import SETTINGS
from .ui import UserInterface
from .video_feed import VideoFeed

import traceback as _traceback

from .modules import run_enabled_modules

def process_frame(frame):
	try:
		run_enabled_modules([frame])
	except Exception:
		print('ERROR: module inference crashed:', flush=True)
		_traceback.print_exc()
	return frame

video_feed = VideoFeed(
	capture_location=SETTINGS.input_source(),
	frametime=1./SETTINGS.FRAMERATE,
	callback=process_frame
)

app = UserInterface(
	title='Test Window',
	video_feed=video_feed
)

app.add_modules(models=LOADED_MODULES)

app.start()
