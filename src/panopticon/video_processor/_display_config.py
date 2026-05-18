"""Display configuration options for the video feed."""

from typing import Literal as _Literal, TypeAlias as _TypeAlias

import ipywidgets as _widgets

from ..typing import FrameCallback

#__all__: list[str] = [x for x in dir() if not x.startswith('_')]
__all__: list[str] = [
	'Headless',
	'OpenCV',
	'Jupyter',
	'Custom',
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

	def __init__(
		self,
		*,
		frametime: float = 1.0 / 30.0
	) -> None:
		self.frametime = frametime


class Jupyter:
	"""Display the video feed in a Jupyter notebook widget.

	Allows the display to work over a remote connection.

	Args
	----
	frametime : float, default=1.0/30.0
		In seconds, how long between each frame. Use `1 / fps` if you want to pass in a framerate.

	format : Literal['jpeg'] | Literal['png'], default='jpeg'
		What image format to convert the frame to.

		Note: JPEG is faster than PNG

	"""

	frametime: float
	image_widget: _widgets.Image

	def __init__(
		self,
		*,
		frametime: float = 1/30,
		format: _Literal['jpeg'] | _Literal['png'] = 'jpeg',
	) -> None:
		self.frametime = frametime
		self.image_widget = _widgets.Image(format=format)


class Custom:
	"""Allow the user to define a way to display the output.

	Args
	----
	func : :func:`FrameCallback`
		The custom display function.

	frametime : float, optional
		In seconds, how long between each frame. Use `1 / fps` if you want to pass in a framerate.
	"""

	func: FrameCallback
	frametime: float

	def __init__(
		self,
		func: FrameCallback,
		*,
		frametime: float = 1/30
	) -> None:
		self.frametime = frametime
		self.func = func


DisplayConfigType: _TypeAlias = Headless | OpenCV | Jupyter | Custom
"""Selector for the type of video display to use."""
