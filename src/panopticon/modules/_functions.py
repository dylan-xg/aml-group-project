"""A collection of functions to loading and using modules."""

# IDEA Could analysis be run asynchronous from the framerate?

from pathlib import Path as _Path
from typing import Iterable as _Iterable

import numpy as _np

from panopticon.drawer import (
	Drawer as _Drawer,
	Text as _Text,
)

from ._base_module import BaseModule as _BaseModule
from ._modules import (
	AntiSpoofModule as AntiSpoofModule,
	EmotionModule as EmotionModule,
)


# A proper implementation will need to be done when we have modules to load.
def _load_all_modules() -> _Iterable[_BaseModule]:
	models: list[_BaseModule] = [
		EmotionModule(
			path=_Path("src/panopticon/model_weights/expression9_orig_longrun.keras")
		).load_model(),
		AntiSpoofModule(
			path=_Path("src/panopticon/model_weights/anti_spoof_model.keras")
		).load_model(),
	]
	return models


LOADED_MODULES: _Iterable[_BaseModule] = _load_all_modules()


def run_enabled_modules(drawer: _Drawer, debug: bool = False) -> None:

	faces_arr = _np.array([face.image for face in drawer.faces])

	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models
		if not model.enabled:
			continue
		if debug:
			print(f"Running module: {model.name}")

		results = model.run_inference(faces=faces_arr)

		for result, face in zip(results, drawer.faces):
			face.texts.append(_Text(result))
