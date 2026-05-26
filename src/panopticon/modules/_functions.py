"""A collection of functions to loading and using modules."""

# TODO Could analysis be run asynchronous from the framerate?

from pathlib import Path as _Path
from typing import Iterable as _Iterable

from ._base_module import BaseModule as _BaseModule
from ._modules import (
	EmptyModule as _EmptyModule,
	KerasModule as _KerasModule,
	EmotionModule as EmotionModule
)
from ..settings import SETTINGS as _SETTINGS


def _fake_load_models() -> _Iterable[_BaseModule]:
	"""Testing function."""

	fake_models: list[_BaseModule] = []
	for i in range(10):
		fake_models.append(_EmptyModule(name=f'Module {i}'))

	return fake_models


# A proper implementation will need to be done when we have modules to load.
def _load_all_modules() -> _Iterable[_BaseModule]:
	models: list[_BaseModule] = [
		EmotionModule(path=_Path('src/panopticon/model_weights/expression9_orig_longrun.keras'))
	]

	for m in models:
		m.load_model()
		print(f'Loaded {m.name}')
	return models


LOADED_MODULES: _Iterable[_BaseModule] = _load_all_modules()

# --- Testing ---

def run_enabled_modules(faces):


	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models
		if not model.enabled: continue
		print(f'Running module: {model.name}')

		result = model.run_inference(faces=faces)

		print(f'{model.name} result:')
		print(result)
