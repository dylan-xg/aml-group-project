"""The entry point for panopticon."""

from .modules import LOADED_MODELS, run_enabled_models
from .settings import SETTINGS
from .ui import UserInterface
from .video_feed import VideoFeed


video_feed = VideoFeed(
	capture_location=SETTINGS.input_source(),
	callback=run_enabled_models,
	frametime=1./SETTINGS.FRAMERATE
)

app = UserInterface(
	title='Test Window',
	video_feed=video_feed
)

app.add_models(models=LOADED_MODELS)

app.start()
