"""A collection of functions to loading and using modules."""

# IDEA Could analysis be run asynchronous from the framerate?
# IDEA Dispatch model inference calls in parallel.

from panopticon.drawer import (
	Drawer as _Drawer,
	Text as _Text,
)
from panopticon.typing import Frame as _Frame

from ._base_module import BaseModule as _BaseModule
from ._modules import (
	AntiSpoofModule as _AntiSpoofModule,
	EmotionModule as _EmotionModule,
	GlassesDetectorModule as _GlassesDetectorModule,
)


def _load_all_modules() -> tuple[_BaseModule, ...]:
	models: list[_BaseModule] = [
		_EmotionModule().load_model(),
		_AntiSpoofModule().load_model(),
		_GlassesDetectorModule().load_model(),
	]
	return tuple(models)


LOADED_MODULES: tuple[_BaseModule, ...] = _load_all_modules()


def run_enabled_modules(drawer: _Drawer, /, *, debug: bool = False) -> None:
	if debug:
		print("Running modules.")

	if len(LOADED_MODULES) == 0:
		if debug:
			print("No modules loaded.")
		return

	images: list[_Frame] = [face.image for face in drawer.faces]
	if len(images) == 0:
		if debug:
			print("No images found.")
		return

	for model in LOADED_MODULES:
		print(f"Module: {model.name}", end="")

		# Skip disabled models.
		if model.enabled is not True:
			if debug:
				print(", skipping")
			continue

		if debug:
			print(", running")

		results: list[str] = model.run_inference(images)

		for result, face in zip(results, drawer.faces):
			face.texts.append(_Text(result))
