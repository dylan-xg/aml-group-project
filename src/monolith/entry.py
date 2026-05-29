from deepface import DeepFace
from deepface.models.facial_recognition.Facenet import (
	load_facenet128d_model,  # LEGACY, for testing if the issue is your model file.
)
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, preprocessing
from deepface.modules.verification import confidences, thresholds
from keras.models import load_model

from .controller import FaceRecognitionController
from .settings import *


# ============================================================
# Custom DeepFace Runtime Extension


def load_my_custom_model():

	model = load_model("my_model.keras")  # put your .keras model file here

	return model


class NewModelClient(FacialRecognition):
	def __init__(self) -> None:
		self.model = load_my_custom_model()
		self.model_name = CUSTOM_MODEL
		self.input_shape = self.model.input_shape[1:3]
		self.output_shape = self.model.output_shape[-1]
		type(self.model)


modeling.AVAILABLE_MODELS["facial_recognition"][CUSTOM_MODEL] = (
	NewModelClient  # adding your model name to the avaliable model list in the right category, facial recognition
)

thresholds[CUSTOM_MODEL] = thresholds[
	"Facenet"
]  # just here because i cloned facenet to test the code, format: (variable) thresholds: dict[str, Any]

confidences[CUSTOM_MODEL] = confidences[
	"Facenet"
]  # just here because i cloned facenet to test the code, format: (variable) confidences: dict[str, dict[str, dict[str, float]]]


original_normalize_input = preprocessing.normalize_input


def custom_normalize_input(img, normalization="base"):

	if normalization == CUSTOM_MODEL:
		# your custom normalization logic, change for each model
		mean, std = img.mean(), img.std()
		img = (img - mean) / std

		# do anything you want within these comments ^
		return img

	return original_normalize_input(img=img, normalization=normalization)


preprocessing.normalize_input = custom_normalize_input

# ============================================================

DeepFace.build_model(RECOGNITION_MODEL)

DeepFace.build_index(RECOGNITION_MODEL)

print("This could take a minute, give it time...")
print("Use the Q key to exit the application.")
FaceRecognitionController().run(source=0)
