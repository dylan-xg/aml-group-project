
from abc import abstractmethod
from typing import Any as _Any

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

	@abstractmethod
	def load_model(self): raise NotImplementedError

	@abstractmethod
	def run_inference(self, frame: Frame) -> _Any:
		raise NotImplementedError


def load_all_models() -> set[BaseModel]:
	return set()

LOADED_MODELS: set[BaseModel] = load_all_models()


# --- If you need to create a class specific to your model, do it here

class ExampleModel(BaseModel):
	def __init__(self, name: str) -> None:
		self.name = name

	def run_inference(self, frame: Frame):
		return False
