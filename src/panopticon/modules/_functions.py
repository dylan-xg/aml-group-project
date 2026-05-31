"""A collection of functions to loading and using modules."""

# IDEA Could analysis be run asynchronous from the framerate?
# IDEA Dispatch model inference calls in parallel.

from typing import Final as _Final

from panopticon.drawer import (
	Drawer as _Drawer,
	Text as _Text,
)
from panopticon.settings import SETTINGS as _SETTINGS
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


LOADED_MODULES: _Final[tuple[_BaseModule, ...]] = _load_all_modules()


def run_enabled_modules(drawer: _Drawer, /) -> None:
	if len(LOADED_MODULES) == 0:
		return

	images: list[_Frame] = [face.image for face in drawer.faces]
	if len(images) == 0:
		return

	for module in LOADED_MODULES:
		# Skip disabled models.
		if module.enabled is not True:
			continue

		results: list[str] = module.run_inference(images)

		for result, face in zip(results, drawer.faces):
			scale: float = 0.4 * _SETTINGS.TEXT_SCALE
			face.texts.append(_Text(label=result, scale=scale))


assert __package__ == "panopticon.modules", (
	f"panopticon.modules imported under wrong package name: {__package__}. "
	"Run with `un run python -m panopticon` or fix debugger launch config."
)
