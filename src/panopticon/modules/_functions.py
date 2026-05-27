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
from ..drawer import Drawer, Face


# A proper implementation will need to be done when we have modules to load.
def _load_all_modules() -> _Iterable[_BaseModule]:
	models: list[_BaseModule] = [
		EmotionModule(path=_Path('src/panopticon/model_weights/expression9_orig_longrun.keras')).load_model()
	]
	return models


LOADED_MODULES: _Iterable[_BaseModule] = _load_all_modules()


def run_enabled_modules(faces):
	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models
		if not model.enabled: continue
		print(f'Running module: {model.name}')

		result = model.run_inference(faces=faces)

		print(f'{model.name} result:')
		print(result)
