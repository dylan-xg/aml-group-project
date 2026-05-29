"""A module is a model that will receive a batch of faces and will provide results depending on its specific implementation."""

from pathlib import Path as _Path
from typing import (
	Self as _Self,
	override as _override,
)

import keras.layers as _layers
import keras.models as _models
import numpy.typing as _npt
import tensorflow as _tf

from ._base_module import (
	BaseModule as _BaseModule,
	kw_dataclass as _kw_dataclass,
)


@_kw_dataclass
class EmptyModule(_BaseModule):
	"""Empty implementation module for testing purposes."""

	@_override
	def load_model(self) -> _Self:
		# Flat example.
		inputs = _layers.Input((1))
		outputs = _layers.Dense(1)(inputs)
		self.model = _models.Model(inputs, outputs)
		return self

	@_override
	def run_inference(self, faces: _npt.NDArray) -> list[str]:
		if self.model is None:
			raise ValueError("Model not loaded")
		results = self.model(faces)
		return [str(r) for r in results.numpy()]


@_kw_dataclass
class EmotionModule(_BaseModule):
	path: _Path
	name: str = "EmotionModule"

	IMG_LENGTH = 128
	IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)

	EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

	@_override
	def load_model(self) -> _Self:
		self.model = _models.load_model(filepath=self.path, compile=False)
		return self

	def preprocess_faces(self, faces):
		# Convert input to tensor
		faces_tensor: _tf.Tensor = _tf.convert_to_tensor(faces)

		if len(faces_tensor.shape) == 3:
			faces_tensor = _tf.expand_dims(faces_tensor, axis=0)

		faces_tensor = faces_tensor[:, ::-1]
		faces_tensor = _tf.image.resize(faces_tensor, self.IMG_SIZE)
		faces_tensor = _tf.cast(faces_tensor, _tf.float32) / 255.0
		return faces_tensor

	@_override
	def run_inference(self, faces: _npt.NDArray) -> list[str]:
		if self.model is None:
			raise ValueError("Model not loaded")
		emotion_faces = self.preprocess_faces(faces)
		results: _tf.Tensor = self.model(emotion_faces, training=False)
		# TODO Use the labels
		return [str(r[0]) for r in results.numpy()]


@_kw_dataclass
class AntiSpoofModule(_BaseModule):
	path: _Path
	name: str = "AntiSpoofModule"

	IMG_LENGTH = 64
	IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)

	@_override
	def load_model(self) -> _Self:

		self.model = _models.load_model(filepath=self.path, compile=False)

		return self

	def preprocess_faces(self, faces):
		faces_tensor: _tf.Tensor = _tf.convert_to_tensor(faces)

		if len(faces_tensor.shape) == 3:
			faces_tensor = _tf.expand_dims(faces_tensor, axis=0)

		faces_tensor = _tf.image.resize(faces_tensor, self.IMG_SIZE)
		faces_tensor = _tf.cast(faces_tensor, _tf.float32) / 255.0
		return faces_tensor

	@_override
	def run_inference(self, faces: _npt.NDArray) -> list[str]:
		if self.model is None:
			raise ValueError("Model not loaded")

		spoof_faces = self.preprocess_faces(faces)
		predictions: _tf.Tensor = self.model(spoof_faces, training=False)
		return [
			f"{self.name}: REAL" if float(pred[0]) >= 0.5 else f"{self.name}: FAKE"
			for pred in predictions.numpy()
		]
