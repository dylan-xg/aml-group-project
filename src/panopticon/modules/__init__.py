"""A module is a model that will receive a batch of faces and will provide results depending on the type of model."""

from ._modules import (
	BaseModel as BaseModel,
	ExampleModel as ExampleModel,
	LOADED_MODELS as LOADED_MODELS,
	run_enabled_models as run_enabled_models
)
