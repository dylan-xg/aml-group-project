"""A module is a model that will receive a batch of faces and will provide results depending on the type of model."""

from abc import abstractmethod as _abstractmethod
from typing import (
	Any as _Any,
	Iterable as _Iterable,
	override as _override
)

import keras

from ..typing import Frame, ModelStateCallback


class BaseModule:
	"""A representational holder class for a model."""
	name: str
	enabled: bool = False
	embedding_model: keras.Model

	def toggle_enabled(
		self,
		callback: ModelStateCallback
	) -> None:
		self.enabled = not self.enabled
		callback(self.enabled)

	@_abstractmethod
	def load_model(self): raise NotImplementedError

	@_abstractmethod
	def run_inference(self, frame: Frame) -> _Any:
		raise NotImplementedError


# --- If you need to create a class specific to your model, do it here ---

class ExampleModule(BaseModule):

	# --- For testing ---
	IMG_LENGTH = 64
	IMG_SIZE = (IMG_LENGTH, IMG_LENGTH)
	IMG_SHAPE = IMG_SIZE + (3,)

	def __init__(self, name: str = '') -> None:
		self.name = name


	@_override
	def load_model(self):
		#preprocess_layer = keras.applications.mobilenet_v2.preprocess_input
		#weights = keras.applications.MobileNetV2(
		#	include_top=False,
		#	input_shape=self.IMG_SHAPE
		#)
		weights = keras.applications.EfficientNetV2B2(
			include_top=False,
			input_shape=self.IMG_SHAPE
		)
		weights.trainable = False
		globalavg_layer = keras.layers.GlobalAveragePooling2D()

		inputs = keras.Input(shape=self.IMG_SHAPE)
		#x: _Any = preprocess_layer(inputs)
		x = weights(inputs, training=False)
		latent_dim: _Any = globalavg_layer(x)
		self.embedding_model: keras.Model = keras.Model(inputs, latent_dim)


	@_override
	def run_inference(self, frame: Frame):
		return False


# --- Module level definitions ---

def _fake_load_models() -> _Iterable[BaseModule]:
	"""Testing function."""

	fake_models: list[BaseModule] = []
	for i in range(10):
		fake_models.append(ExampleModule(name=f'Model {i}'))

	return fake_models


# A proper implementation will need to be done when we have modules to load.
def _load_all_modules() -> _Iterable[BaseModule]:
	models = _fake_load_models()
	for m in models:
		print(f'Loaded {m.name}')
	return models


LOADED_MODULES: _Iterable[BaseModule] = _load_all_modules()

# --- Testing ---

def run_enabled_modules(frame: Frame):
	# This could be run asynchronous from the framerate.
	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models
		if not model.enabled: continue
		model.run_inference(frame=frame)
