"""Facial recognition embedding, detection, verification, and extraction."""

from ._comparison import (
	compare_faces as compare_faces,
	compare_faces_from_path as compare_faces_from_path,
)
from ._deepface import add_model_to_deepface as add_model_to_deepface
from ._model import (
	CustomClassifierEmbeddingModel as CustomClassifierEmbeddingModel,
	ExampleModel as ExampleModel,
    MetricLearningEmbeddingModel,
    BaseDeepFaceEmbeddingModel,
)
from ._registration import NewFace as NewFace
from ._run import FaceRecognitionSystem as FaceRecognitionSystem
