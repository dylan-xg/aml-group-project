"""A module is a model that will receive a batch of faces and will provide
results depending on its specific implementation."""

from pathlib import Path as _Path
from typing import (
	Self as _Self,
	override as _override,
)

import keras.layers as _layers
import keras.models as _models
import numpy.typing as _npt
import tensorflow as _tf

from panopticon.settings import SETTINGS as _SETTINGS
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
	def run_inference(self, face_list: list[_Frame], /) -> list[str]:
		results_tensor: _tf.Tensor = self._call_model(face_list)
		return [str(r) for r in results_tensor.numpy()]


@_kw_dataclass
class EmotionModule(_BaseModule):
	name: str = "EmotionModule"
	path: _Path = _SETTINGS.MODEL_WEIGHTS_LOCATION / "expression9_orig_longrun.keras"
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
	def run_inference(self, face_list: list[_Frame], /) -> list[str]:
		results_tensor: _tf.Tensor = self._call_model(face_list)
		return [
			self.EMOTION_LABELS[prediction.argmax()]
			for prediction in results_tensor.numpy()
		]


@_kw_dataclass
class AntiSpoofModule(_BaseModule):
	name: str = "AntiSpoofModule"
	path: _Path = _SETTINGS.MODEL_WEIGHTS_LOCATION / "anti_spoof_model.keras"
	img_length: int = 64
	img_size: tuple[int, int] = (img_length, img_length)

	@_override
	def run_inference(self, face_list: list[_Frame], /) -> list[str]:
		results_tensor: _tf.Tensor = self._call_model(face_list, flip_channels=False)
		return [
			f"{self.name}: REAL" if float(pred[0]) >= 0.9 else f"{self.name}: FAKE"
			for pred in results_tensor.numpy()
		]


@_kw_dataclass
class GlassesDetectorModule(_BaseModule):
	name: str = "Glasses"
	path: _Path = _SETTINGS.MODEL_WEIGHTS_LOCATION / "glasses_detection.keras"
	img_length: int = 64
	img_size: tuple[int, int] = (img_length, img_length)

	@_override
	def run_inference(self, face_list: list[_Frame], /) -> list[str]:
		results_tensor: _tf.Tensor = self._call_model(
			face_list, normalise=False, convert_to_uint8=True
		)
		results: _npt.NDArray = results_tensor.numpy().flatten() > 0
		return [
			f"{self.name}: yes" if result else f"{self.name}: no"
			for result in results.tolist()
		]
