
from abc import abstractmethod as _abstractmethod
from typing import Any as _Any, Iterable as _Iterable

from ..typing import Frame, ModelStateCallback


class BaseModel:
	"""A representational holder class for a model."""
	name: str
	enabled: bool = False

	def toggle_enabled(
		self,
		callback: ModelStateCallback
	) -> None:
		self.enabled = not self.enabled
		callback(self.enabled)

	@_abstractmethod
	def load_model(self): raise NotImplementedError

	@_abstractmethod
	def run_inference(self, frame: Frame) -> _Any:
		raise NotImplementedError


# --- If you need to create a class specific to your model, do it here ---

class ExampleModel(BaseModel):
	def __init__(self, name: str) -> None:
		self.name = name

	def load_model(self):
		pass

	def run_inference(self, frame: Frame):
		return False


# --- Module level definitions ---

def _fake_load_models() -> _Iterable[BaseModel]:
	"""Testing function."""

	fake_models: list[BaseModel] = []
	for i in range(10):
		fake_models.append(ExampleModel(name=f'Model {i}'))

	return fake_models


def _load_all_models() -> _Iterable[BaseModel]:
	models = _fake_load_models()
	for m in models:
		print(f'Loaded {m.name}')
	return models


LOADED_MODELS: _Iterable[BaseModel] = _load_all_models()

# --- Testing ---

def run_enabled_models(frame: Frame):
	# This could be run asynchronous from the framerate.
	# Could maybe dispatch model inference calls in parallel.
	for model in LOADED_MODELS:
		# Skip disabled models
		if not model.enabled: continue
		model.run_inference(frame=frame)
