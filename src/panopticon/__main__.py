"""The entry point for panopticon."""

from .modules import LOADED_MODULES
from .settings import SETTINGS
from .ui import UserInterface
from .video_feed import VideoFeed


video_feed = VideoFeed(
	capture_location=SETTINGS.input_source(),
	frametime=1./SETTINGS.FRAMERATE
)

app = UserInterface(
	title='Test Window',
	video_feed=video_feed
)

app.add_modules(models=LOADED_MODULES)

app.start()
