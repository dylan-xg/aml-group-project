"""A module is a model that will receive a batch of faces and will provide results depending on the type of model."""

from ._modules import (
	BaseModule as BaseModule,
	ExampleModule as ExampleModule,
	LOADED_MODULES as LOADED_MODULES,
	run_enabled_modules as run_enabled_modules
)
