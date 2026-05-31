"""An abstract base class for the modules to inherit from."""

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from collections.abc import Callable as _Callable
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path
from typing import (
	Self as _Self,
	dataclass_transform as _dataclass_transform,
)

import keras.models as _models
import tensorflow as _tf
from keras.src import Functional as _Functional

from panopticon.typing import (
	Frame as _Frame,
	ModuleStateCallback as _ModuleStateCallback,
)


@_dataclass_transform(kw_only_default=True)
def kw_dataclass[T](
	cls: type[T] | None = None, **kwargs
) -> type[T] | _Callable[[type[T]], type[T]]:
	"""Custom decorator shorthand for @dataclass(kw_only=True)."""
	# Force kw_only to True.
	kwargs["kw_only"] = True

	# Handle the case where the decorator is called without parentheses: @kw_dataclass
	if cls is None:

		def wrapper(c: type[T]) -> type[T]:
			return _dataclass(c, **kwargs)

		return wrapper

	# Handle the case where the decorator is called with parentheses: @kw_dataclass()
	return _dataclass(cls, **kwargs)


@kw_dataclass
class BaseModule(_ABC):
	"""A representational holder class for a model."""

	name: str
	img_length: int
	img_size: tuple[int, int]
	"""Don't set directly"""
	path: _Path
	enabled: bool = False
	# Prevent this from being included in the dataclass initialiser.
	model: _Functional | None = _field(default=None, init=False)

	def toggle_enabled(self, callback: _ModuleStateCallback, /) -> None:
		self.enabled = not self.enabled
		callback(self.enabled)

	def load_model(self) -> _Self:
		"""A method to load the model weights for this module."""
		self.model = _models.load_model(filepath=self.path, compile=False)
		return self

	def preprocess_faces(
		self,
		face_list: list[_Frame],
		/,
		flip_channels=True,
		normalise=True,
		convert_to_uint8=False,
	) -> _tf.Tensor:
		processed_faces: list[_tf.Tensor] = []

		# Resize individually.
		for face in face_list:
			tensor: _tf.Tensor = _tf.convert_to_tensor(face)

			if len(tensor.shape) == 2:
				tensor = _tf.expand_dims(tensor, axis=-1)

			tensor = _tf.image.resize(tensor, self.img_size)
			processed_faces.append(tensor)

		# Batch the images as (Batch, Height, Width, Channels).
		faces_tensor: _tf.Tensor = _tf.stack(processed_faces)

		if flip_channels:
			# This is effectively the same as faces_tensor[..., ::-1].
			faces_tensor = _tf.reverse(faces_tensor, axis=[-1])

		if normalise:
			faces_tensor = _tf.cast(faces_tensor, _tf.float32) / 255.0

		if convert_to_uint8:
			faces_tensor = _tf.cast(faces_tensor, _tf.uint8)

		return faces_tensor

	def _call_model(self, face_list: list[_Frame], /, **kwargs) -> _tf.Tensor:
		"""Helper function for :func:`run_inference`."""
		if self.model is None:
			raise RuntimeError("Model not loaded.")

		if len(face_list) == 0:
			raise ValueError("Empty list passed.")

		tensor: _tf.Tensor = self.preprocess_faces(face_list, **kwargs)
		return self.model(tensor, training=False)

	@_abstractmethod
	def run_inference(self, face_list: list[_Frame], /) -> list[str]:
		"""A method to run model inference on the passed faces.

		Parameters
		----------
		faces : list of images
			These are the different faces detected this frame that will be run on.

		Returns
		-------
		results : list of strings
			Matching 1:1 to the images in the input, the result from the model.
		"""
		raise NotImplementedError
