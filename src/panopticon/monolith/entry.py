
from pathlib import Path

from deepface import DeepFace
from deepface.modules import preprocessing, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.models.facial_recognition.Facenet import (load_facenet128d_model) #LEGACY, for testing if the issue is your model file.
from deepface.modules.verification import thresholds, confidences
from keras.models import load_model

from settings import *
from .controller import FaceRecognitionController

# ============================================================
# Custom DeepFace Runtime Extension

def load_my_custom_model():

	model = load_model("my_model.keras") #put your .keras model file here

	return model

class NewModelClient(FacialRecognition):
	def __init__(self) -> None:
		self.model = load_my_custom_model()
		self.model_name = CUSTOM_MODEL
		self.input_shape = self.model.input_shape[1:3]
		self.output_shape = self.model.output_shape[-1]
		type(self.model)


modeling.AVAILABLE_MODELS["facial_recognition"][CUSTOM_MODEL] = NewModelClient #adding your model name to the avaliable model list in the right category, facial recognition

thresholds[CUSTOM_MODEL] = thresholds["Facenet"] #just here because i cloned facenet to test the code, format: (variable) thresholds: dict[str, Any]

confidences[CUSTOM_MODEL] = confidences["Facenet"] #just here because i cloned facenet to test the code, format: (variable) confidences: dict[str, dict[str, dict[str, float]]]


original_normalize_input = preprocessing.normalize_input


def custom_normalize_input(img, normalization="base"):

	if normalization == CUSTOM_MODEL:

		# your custom normalization logic, change for each model
		mean, std = img.mean(), img.std()
		img = (img - mean) / std

		#do anything you want within these comments ^
		return img

	return original_normalize_input(img=img,normalization=normalization)

preprocessing.normalize_input = custom_normalize_input

# ============================================================

DeepFace.build_model(RECOGNITION_MODEL)

def convert_path_to_string(path: Path, /) -> str:
	all_suffixes = ''.join(path.suffixes)
	base_name = path.name.removesuffix(all_suffixes)
	return '_'.join(path.parts[:-1] + (base_name,))


def register(
	path: Path | list[Path],
	/,
	model: str = RECOGNITION_MODEL
) -> int:
	if not isinstance(path, list):
		path = [path]

	sum = 0

	for p in path:
		result: dict[str, int] = DeepFace.register(
			img=str(p),
			img_name=convert_path_to_string(p),
			model_name=model,
			normalization=model,
			enforce_detection=False #changed
		)
		sum += result['inserted']

	return sum


DeepFace.build_index(RECOGNITION_MODEL)

print("This could take a minute, give it time...")
print("Use the Q key to exit the application.")
FaceRecognitionController().run(VIDEO_SOURCE=VIDEO_SOURCE)
