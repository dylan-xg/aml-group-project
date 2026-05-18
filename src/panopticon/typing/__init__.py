"""Types are good."""

from ._types import (
	Frame as Frame,
	ProcessFrameCallback as ProcessFrameCallback,
	FrameCallback as FrameCallback
)

__all__: list[str] = ['Frame', 'ProcessFrameCallback', 'FrameCallback']
#__all__: list[str] = list(set(_types.__all__))
