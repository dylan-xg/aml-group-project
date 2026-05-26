
# TODO Could analysis be run asynchronous from the framerate?

from typing import Iterable as _Iterable


from ._base_module import BaseModule
from ._modules import EmptyModule


# --- Module level definitions ---

def _fake_load_models() -> _Iterable[BaseModule]:
	"""Testing function."""

	fake_models: list[BaseModule] = []
	for i in range(10):
		fake_models.append(EmptyModule(name=f'Module {i}'))

	return fake_models


# A proper implementation will need to be done when we have modules to load.
def _load_all_modules() -> _Iterable[BaseModule]:
	models = _fake_load_models()
	for m in models:
		print(f'Loaded {m.name}')
	return models


LOADED_MODULES: _Iterable[BaseModule] = _load_all_modules()

# --- Testing ---

def run_enabled_modules(faces):
	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODULES:
		# Skip disabled models
		if not model.enabled: continue
		model.run_inference(faces=faces)
