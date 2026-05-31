"""The entry point for panopticon."""

from .modules import LOADED_MODULES
from .recognition import (
	#CustomClassifierEmbeddingModel,
	FaceRecognitionSystem,
	MetricLearningEmbeddingModel,
	add_model_to_deepface,
)
from .settings import SETTINGS
from .ui import UserInterface
from .video_feed import VideoFeed


model_path = (
	SETTINGS.MODEL_WEIGHTS_LOCATION
	/ "metric_learning_current_after_extra_training.keras"
)
add_model_to_deepface(
	MetricLearningEmbeddingModel(name=SETTINGS.MODEL_NAME, model_path=model_path)
)

video_feed = VideoFeed(
	capture_location=SETTINGS.input_source(),
	callback=FaceRecognitionSystem.detect_in_frame,
	frametime=1.0 / SETTINGS.FRAMERATE,
)

app = UserInterface(title="Test Window", video_feed=video_feed)

app.add_modules(LOADED_MODULES)

app.start()
