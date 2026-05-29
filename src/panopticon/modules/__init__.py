"""A module is a model that will receive a batch of faces and will provide results depending on the type of model."""

from ._base_module import BaseModule as BaseModule
from ._functions import (
	LOADED_MODULES as LOADED_MODULES,
	run_enabled_modules as run_enabled_modules,
)
from ._modules import (
	EmotionModule as EmotionModule,
	ExampleModule as ExampleModule,
	KerasModule as KerasModule,
)
