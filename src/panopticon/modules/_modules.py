"""A module is a model that will receive a batch of faces and will provide
results depending on its specific implementation."""

from typing import (
	Self as _Self,
	override as _override,
)

import keras.layers as _layers
import keras.models as _models
import tensorflow as _tf

from panopticon.typing import Frame as _Frame

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
	def run_inference(self, faces: list[_Frame]) -> list[str]:
		if self.model is None:
			raise ValueError("Model not loaded")

		if len(faces) == 0:
			return []

		tensor = self.preprocess_faces(faces)
		results = self.model(tensor)
		return [str(r) for r in results.numpy()]


@_kw_dataclass
class EmotionModule(_BaseModule):
	name: str = "EmotionModule"
	img_length: int = 128
	img_size: tuple[int, int] = (img_length, img_length)

	EMOTION_LABELS = [
		"Angry",
		"Contempt",
		"Disgust",
		"Fear",
		"Happy",
		"Natural",
		"Sad",
		"Sleepy",
		"Surprised",
	]

	@_override
	def run_inference(self, faces: list[_Frame]) -> list[str]:
		if self.model is None:
			raise ValueError("Model not loaded")

		if len(faces) == 0:
			return []

		emotion_faces = self.preprocess_faces(faces)
		results: _tf.Tensor = self.model(emotion_faces, training=False)

		return [
			self.EMOTION_LABELS[prediction.argmax()] for prediction in results.numpy()
		]


@_kw_dataclass
class AntiSpoofModule(_BaseModule):
	name: str = "AntiSpoofModule"
	img_length: int = 64
	img_size: tuple[int, int] = (img_length, img_length)

	@_override
	def run_inference(self, faces: list[_Frame]) -> list[str]:
		if self.model is None:
			raise ValueError("Model not loaded")

		if len(faces) == 0:
			return []

		spoof_faces = self.preprocess_faces(faces)
		predictions: _tf.Tensor = self.model(spoof_faces, training=False)

		return [
			f"{self.name}: REAL" if float(pred[0]) >= 0.5 else f"{self.name}: FAKE"
			for pred in predictions.numpy()
		]
