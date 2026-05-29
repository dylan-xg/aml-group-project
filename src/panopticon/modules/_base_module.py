"""An abstract base class for the modules to inherit from."""

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from dataclasses import dataclass as _dataclass, field as _field
from typing import (
	Callable as _Callable,
	Self as _Self,
	dataclass_transform as _dataclass_transform,
)

import numpy.typing as _npt
from keras.src import Functional as _Functional

from panopticon.typing import ModuleStateCallback as _ModuleStateCallback


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
	enabled: bool = False
	# Prevent this from being included in the dataclass initialiser.
	model: _Functional | None = _field(default=None, init=False)

	def toggle_enabled(self, callback: _ModuleStateCallback) -> None:
		self.enabled = not self.enabled
		callback(self.enabled)

	@_abstractmethod
	def load_model(self) -> _Self:
		"""A method to load the model weights for this module."""
		raise NotImplementedError

	@_abstractmethod
	def run_inference(self, faces: _npt.NDArray) -> list[str]:
		"""A method to run model inference on the passed faces.

		Parameters
		----------
		faces : images containing faces
			This is a 4D numpy array.

			The dimensions are as follows: (n, height, width, 3)

			Where n is the number of faces in the array.

		Returns
		-------
		results : list of strings
			Matching 1:1 to the images in the input, the result from the model.
		"""
		raise NotImplementedError
