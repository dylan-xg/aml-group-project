"""A module is a model that will receive a batch of faces and will provide results depending on its specific implementation."""

from dataclasses import (
	field as _field
)
from pathlib import Path as _Path
from typing import (
	override as _override,
	Self as _Self
)

import keras.layers as _layers
import keras.models as _models
import keras.applications as _apps
from keras.src import Functional as _Functional

import tensorflow as _tf

from ._base_module import (
	BaseModule as _BaseModule,
	kw_dataclass as _kw_dataclass
)


@_kw_dataclass
class EmptyModule(_BaseModule):
	"""Empty implementation module for testing purposes."""

	@_override
	def load_model(self) -> None: pass


	@_override
	def run_inference(self, faces) -> str: return ''


@_kw_dataclass
class ExampleModule(_BaseModule):
	"""Example implementation module for testing purposes."""

	name: str = 'ExampleModule'
	IMG_LENGTH = 64
	IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)
	IMG_SHAPE = IMG_SIZE + (3,)


	@_override
	def load_model(self) -> None:
		#preprocess_layer = _apps.mobilenet_v2.preprocess_input
		#weights = _apps.MobileNetV2(
		#	include_top=False,
		#	input_shape=self.IMG_SHAPE
		#)
		weights: _Functional = _apps.EfficientNetV2B2(
			include_top=False,
			input_shape=self.IMG_SHAPE
		)
		weights.trainable = False
		globalavg_layer = _layers.GlobalAveragePooling2D()

		inputs = _layers.Input(shape=self.IMG_SHAPE)
		#x: _Any = preprocess_layer(inputs)
		x = weights(inputs, training=False)
		latent_dim = globalavg_layer(x)
		self.model = _Functional(inputs=inputs, outputs=latent_dim, trainable=False)


	@_override
	def run_inference(self, faces) -> str:
		if self.model is None: raise ValueError('Model not loaded')
		return self.model(faces, training=False)


@_kw_dataclass
class KerasModule(_BaseModule):
	path: _Path
	name: str = _field(init=False)

	def __post_init__(self) -> None:
		self.name = self.path.stem


	@_override
	def load_model(self) -> _Self:
		self.model: _Functional | None = _models.load_model(filepath=self.path, compile=False) # type: ignore
		return self


	@_override
	def run_inference(self, faces) -> str:
		if self.model is None: raise ValueError('Model not loaded')
		return self.model(faces, training=False)


@_kw_dataclass
class EmotionModule(_BaseModule):

	path: _Path
	name: str = 'EmotionModule'

	IMG_LENGTH = 128
	IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)

	EMOTION_LABELS = [
		'Angry',
		'Disgust',
		'Fear',
		'Happy',
		'Sad',
		'Surprise',
		'Neutral'
	]


	@_override
	def load_model(self) -> _Self:
		self.model = _models.load_model(
			filepath=self.path,
			compile=False
		)
		return self


	def preprocess_faces(self, faces):
		# Convert input to tensor
		faces_tensor: _tf.Tensor = _tf.convert_to_tensor(faces)

		if len(faces_tensor.shape) == 3:
			faces_tensor = _tf.expand_dims(faces_tensor, axis=0)

		faces_tensor = faces_tensor[:,::-1]
		faces_tensor = _tf.image.resize(faces_tensor, self.IMG_SIZE)
		faces_tensor = _tf.cast(faces_tensor, _tf.float32) / 255.0
		return faces_tensor


	@_override
	def run_inference(self, faces) -> str:
		if self.model is None:
			raise ValueError('Model not loaded')
		emotion_faces = self.preprocess_faces(faces)
		predictions = self.model(emotion_faces, training=False)

		return predictions
