"""Types are good."""

from ._types import (
	Frame as Frame,
	ProcessFrameCallback as ProcessFrameCallback
)

__all__: list[str] = ['Frame', 'ProcessFrameCallback']
#__all__: list[str] = list(set(_types.__all__))
