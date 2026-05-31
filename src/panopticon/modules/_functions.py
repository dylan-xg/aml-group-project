"""A collection of functions to loading and using modules."""

# IDEA Could analysis be run asynchronous from the framerate?

from pathlib import Path as _Path

from panopticon.drawer import (
	Drawer as _Drawer,
	Text as _Text,
)
from panopticon.typing import Frame as _Frame

from ._base_module import BaseModule as _BaseModule
from ._modules import (
	AntiSpoofModule as AntiSpoofModule,
	EmotionModule as EmotionModule,
)


def _load_all_modules() -> tuple[_BaseModule, ...]:
	models: list[_BaseModule] = [
		EmotionModule(
			path=_Path("src/panopticon/model_weights/expression9_orig_longrun.keras")
		).load_model(),
		AntiSpoofModule(
			path=_Path("src/panopticon/model_weights/anti_spoof_model.keras")
		).load_model(),
	]
	return tuple(models)


LOADED_MODULES: tuple[_BaseModule, ...] = _load_all_modules()


def run_enabled_modules(drawer: _Drawer, debug: bool = False) -> None:
	if len(LOADED_MODULES) == 0:
		return

	images: list[_Frame] = [face.image for face in drawer.faces]
	if len(images) == 0:
		return

	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models.
		if not model.enabled:
			continue

		if debug:
			print(f"Running module: {model.name}")

		results: list[str] = model.run_inference(faces=images)

		for result, face in zip(results, drawer.faces):
			face.texts.append(_Text(result))
