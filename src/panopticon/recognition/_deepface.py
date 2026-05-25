"""Integrates the face recognition model into DeepFace.

It should detect all faces, recognise known faces, and locate them within the frame.
"""

#from deepface import DeepFace
from deepface.modules import preprocessing, modeling
from deepface.models.FacialRecognition import FacialRecognition
#from deepface.models.facial_recognition.Facenet import load_facenet128d_model
from deepface.modules.verification import thresholds, confidences


def add_model_to_deepface(model) -> None:
	class NewModelClient(FacialRecognition):
		def __init__(self) -> None:
			self.model = model.embedding_model
			self.model_name = model.name
			self.input_shape = self.model.input_shape[1:3]
			self.output_shape = self.model.output_shape[-1]
			#type(self.model)

	# Adding your model name to the avaliable model list in the right category, facial recognition.
	modeling.AVAILABLE_MODELS["facial_recognition"][model.name] = NewModelClient

	# Just here because I cloned facenet to test the code.
	# Format: (variable) thresholds: dict[str, Any]
	thresholds[model.name] = thresholds["Facenet"]
	# Format: (variable) confidences: dict[str, dict[str, dict[str, float]]]
	confidences[model.name] = confidences["Facenet"]

def custom_normalisation():
	original_normalise_input = preprocessing.normalize_input

	def custom_normalise_input(img, normalisation="base"):
		if normalisation == 'model.name': # Yeah I know, will be changed
			# --- Custom normalisation logic, change for each model. ---
			mean, std = img.mean(), img.std()
			img = (img - mean) / std
			# --- Do anything you want within these comments. ---
			return img

		return original_normalise_input(img=img, normalization=normalisation)

	preprocessing.normalize_input = custom_normalise_input
