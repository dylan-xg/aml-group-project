"""Display configuration options for the video feed."""

from typing import Literal

import ipywidgets

from .settings import Frame, FrameDisplayCallback


class Headless:
	"""Do not display any type of video output.

	Good for testing.
	"""


class OpenCV:
	"""Use the native OpenCV display output.

	Parameters
	----------
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

	Parameters
	----------
	frametime : float, default=1.0/30.0
		In seconds, how long between each frame. Use `1 / fps` if you want to pass in a framerate.

	format : Literal['jpeg'] | Literal['png'], default='jpeg'
		What image format to convert the frame to.

		Note: JPEG is faster than PNG

	"""

	frametime: float
	image_widget: ipywidgets.Image

	def __init__(
		self,
		*,
		frametime: float = 1/30,
		format: Literal['jpeg'] | Literal['png'] = 'jpeg',
	) -> None:
		self.frametime = frametime
		self.image_widget = ipywidgets.Image(format=format)


class Custom:
	"""Allow the user to define a way to display the output.

	Parameters
	----------
	func : :func:`FrameCallback`
		The custom display function.

	frametime : float, optional
		In seconds, how long between each frame. Use `1 / fps` if you want to pass in a framerate.
	"""

	func: FrameDisplayCallback
	frametime: float

	def __init__(
		self,
		func: FrameDisplayCallback,
		*,
		frametime: float = 1/30
	) -> None:
		self.frametime = frametime
		self.func = func


type DisplayConfigType = Headless | OpenCV | Jupyter | Custom
"""Selector for the type of video display to use."""
