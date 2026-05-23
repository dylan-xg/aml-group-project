# We will need to discuss design decisions regarding how the model should be loaded, stored, and called for inference.

from ._model import Model as Model

def fake_load_models() -> tuple[Model]:
	"""Testing function."""

	fake_models = []
	for i in range(10):
		fake_models.append(Model(f'Model {i}'))

	return tuple(fake_models)
