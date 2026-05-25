"""The entry point for panopticon."""

from src.panopticon.recognition import LOADED_MODELS, run_enabled_models
from src.panopticon.settings import SETTINGS
from src.panopticon.ui import UserInterface
from src.panopticon.video_feed import VideoFeed


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
