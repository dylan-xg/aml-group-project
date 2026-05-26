"""A collection of functions to loading and using modules."""

# TODO Could analysis be run asynchronous from the framerate?

from pathlib import Path as _Path
from typing import Iterable as _Iterable

from ._base_module import BaseModule as _BaseModule
from ._modules import (
	EmptyModule as _EmptyModule,
	KerasModule as _KerasModule
)
from ..settings import SETTINGS as _SETTINGS


def _fake_load_models() -> _Iterable[_BaseModule]:
	"""Testing function."""

	fake_models: list[_BaseModule] = []
	for i in range(10):
		fake_models.append(_EmptyModule(name=f'Module {i}'))

	return fake_models


def _load_all_modules(
	path: _Path = _SETTINGS.MODEL_WEIGHTS_LOCATION,
	pattern: str = '*.keras'
) -> _Iterable[_BaseModule]:
	"""Explore the given directory for files matching a pattern."""
	modules = [
		_KerasModule(path=path).load_model()
		for path in path.glob(pattern=pattern)
	]
	return modules


# Can't use set because it requires the elements to be hashable.
LOADED_MODULES: _Iterable[_BaseModule] = _load_all_modules()


def run_enabled_modules(faces):
	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models
		if not model.enabled: continue
		model.run_inference(faces=faces)
