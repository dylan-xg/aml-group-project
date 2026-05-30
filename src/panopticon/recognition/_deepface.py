"""Integrates the face recognition model into DeepFace.

It should detect all faces, recognise known faces, and locate them within the frame.
"""

import numpy as _np
from deepface import DeepFace as _DeepFace
from deepface.models.FacialRecognition import FacialRecognition as _FacialRecognition
from deepface.modules import (
	modeling as _modeling,
	preprocessing as _preprocessing,
)
from deepface.modules.verification import (
	confidences as _confidences,
	thresholds as _thresholds,
)

from ._model import CustomClassifierEmbeddingModel as _CustomClassifierEmbeddingModel


def add_model_to_deepface(model: _CustomClassifierEmbeddingModel, /) -> None:

	class NewModelClient(_FacialRecognition):
		def __init__(self) -> None:
			self.model = model.embedding_model
			self.model_name = model.name
			self.input_shape = model.input_shape  # type: ignore
			self.output_shape = model.output_shape

		def forward(self, img):
			if img.ndim == 3:
				img = _np.expand_dims(img, axis=0)

			if img.ndim != 4:
				raise ValueError(
					f"Input image must be shaped like (batch, X, X, 3), "
					f"but got {img.shape}"
				)

			embeddings = self.model(img, training=False)

			if hasattr(embeddings, "numpy"):
				embeddings = embeddings.numpy()  # type: ignore

			if embeddings.shape[0] == 1:  # type: ignore
				return embeddings[0].tolist()  # type: ignore

			return embeddings.tolist()  # type: ignore

	_modeling.AVAILABLE_MODELS["facial_recognition"][model.name] = NewModelClient

	_thresholds[model.name] = model.thresholds
	_confidences.update(model.confidences)

	original_normalise_input = _preprocessing.normalize_input

	def custom_normalize_input(img, normalization="base"):
		if normalization == model.normalization:
			return model.normalize(img)

		return original_normalise_input(img=img, normalization=normalization)

	_preprocessing.normalize_input = custom_normalize_input

	_DeepFace.build_model(model_name=model.name)


##def add_example_model_to_deepface(model) -> None:
##	class NewModelClient(_FacialRecognition):
##		def __init__(self) -> None:
##			self.model = model.embedding_model
##			self.model_name = model.name
##			self.input_shape = self.model.input_shape[1:3]
##			self.output_shape = self.model.output_shape[-1]
##			# type(self.model)
##
##	# Adding your model name to the avaliable model list in the right category, facial recognition.
##	_modeling.AVAILABLE_MODELS["facial_recognition"][model.name] = NewModelClient
##
##	# Just here because I cloned facenet to test the code.
##	# Format: (variable) thresholds: dict[str, Any]
##	_thresholds[model.name] = _thresholds["Facenet"]
##	# Format: (variable) confidences: dict[str, dict[str, dict[str, float]]]
##	_confidences[model.name] = _confidences["Facenet"]
##
##
##def example_custom_normalisation():
##	original_normalise_input = _preprocessing.normalize_input
##
##	def example_custom_normalise_input(img, normalisation="base"):
##		if normalisation == "model.name":  # Yeah I know, will be changed
##			# --- Custom normalisation logic, change for each model. ---
##			mean, std = img.mean(), img.std()
##			img = (img - mean) / std
##			# --- Do anything you want within these comments. ---
##			return img
##
##		return original_normalise_input(img=img, normalization=normalisation)
##
##	_preprocessing.normalize_input = example_custom_normalise_input
