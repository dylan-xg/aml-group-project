
from ._models import BaseModel, ExampleModel


def fake_load_models() -> set[BaseModel]:
	"""Testing function."""

	fake_models: set[BaseModel] = set()
	for i in range(10):
		fake_models.add(ExampleModel(name=f'Model {i}'))

	return fake_models
