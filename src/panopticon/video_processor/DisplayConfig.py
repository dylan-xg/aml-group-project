"""Display configuration options for the video feed."""

from typing import Literal as _Literal, TypeAlias as _TypeAlias

import ipywidgets as _widgets

#__all__: list[str] = [x for x in dir() if not x.startswith('_')]
__all__: list[str] = [
	'Headless',
	'OpenCV',
	'Jupyter',
	'DisplayConfigType'
]

class Headless:
	"""Do not display any type of video output.

	Good for testing.
	"""


class OpenCV:
	"""Use the native OpenCV display output.

	Args
	----
	frametime : float, default=1.0/30.0
		In seconds, how long between each frame. Use `1 / fps` if you want to pass in a framerate.
	"""
	frametime: float

	def __init__(self, frametime: float = 1.0 / 30.0) -> None:
		self.frametime = frametime


class Jupyter:
	"""Display the video feed in a Jupyter notebook widget.

	Allows the display to work over a remote connection.

	Args
	----
	format : Literal['jpeg'] | Literal['png']
		JPEG is faster than PNG

	frametime : float, default=1.0/30.0
		In seconds, how long between each frame. Use `1 / fps` if you want to pass in a framerate.
	"""
	image_widget: _widgets.Image
	frametime: float

	def __init__(
		self,
		format: _Literal['jpeg'] | _Literal['png'] = 'jpeg',
		frametime: float = 1/30
	) -> None:
		self.image_widget = _widgets.Image(format=format)
		self.frametime = frametime


DisplayConfigType: _TypeAlias = Headless | OpenCV | Jupyter
"""Selector for the type of video display to use."""
