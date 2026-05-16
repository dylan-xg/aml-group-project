"""Provides a collection of functions for video input and video display."""

import time
from collections.abc import Callable
from typing import TypeAlias
from pathlib import Path

import cv2 as cv
import numpy as np
import numpy.typing as npt
from IPython.display import display

from ..typing import Frame
from . import DisplayConfig
from .DisplayConfig import DisplayConfigType as ConfigType


FrameCallback: TypeAlias = Callable[[Frame], Frame | None]
"""The callback function type."""

_END_LOOP = False
_GO_AGAIN = True

def _display_video(
	display_option: ConfigType,
	frame: Frame
) -> bool:
	"""Will attempt to display an output based on the display option.

	Private function, should not be called directly.
	"""

	match display_option:
		case DisplayConfig.Jupyter(image_widget=widget):
			ret: bool
			buffer: npt.NDArray[np.uint8]
			ret, buffer = cv.imencode(ext='.jpg', img=frame)

			if not ret:
				print("Can't encode frame as image. Exiting ...")
				return _END_LOOP

			widget.value = buffer.tobytes()

		case DisplayConfig.OpenCV():
			cv.imshow(winname='frame', mat=frame)

		case _: # DisplayConfig.Headless:
			pass

	return _GO_AGAIN


def _frame_loop(
	vid_cap: cv.VideoCapture,
	callback: FrameCallback | None,
	display_option: ConfigType,
) -> bool:
	"""This function is called continuously until either the video ends, or is interrupted.

	Private function, should not be called directly.
	"""

	start_time: float = time.time()

	ret: bool
	frame: Frame
	ret, frame = vid_cap.read()

	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		return _END_LOOP

	frame_result: Frame | None = callback(frame) if callback is not None else None
	if frame_result is None: frame_result = frame

	if not _display_video(
		display_option=display_option,
		frame=frame_result
	): return _END_LOOP

	match display_option:
		case DisplayConfig.OpenCV(frametime=ft):
			# Frametime is in seconds, waitKey expects milliseconds
			if cv.waitKey(delay=int(ft * 1000)) == ord('q'):
				return _END_LOOP

		case DisplayConfig.Jupyter(frametime=ft):
			# Enforce the framerate pacing
			elapsed_time: float = time.time() - start_time
			time_to_wait: float = ft - elapsed_time
			if time_to_wait > 0: time.sleep(time_to_wait)

		case _: # DisplayConfig.Headless:
			pass

	return _GO_AGAIN


def process_video(
	capture_location: int | str | Path,
	callback: FrameCallback | None = None,
	display_config: ConfigType | None = None
) -> None:
	"""Read from a video input and apply the callback to it, then optionally display it.

	Args
	----
	capture_location : int | str | Path
		What the video source is.

		- For a webcam input, use an `int`, 0 is the default webcam.
		- For a video file, pass in a file location either as `str` or `Path` (preferred).

	callback : :func:`FrameCallback`, optional
		A function that will be called every frame with every frame.

		:func:`FrameCallback` = `(Frame) -> Frame | None`.

		:func:`Frame` = `cv.typing.MatLike`

	display_config : :func:`DisplayConfig.ConfigType`, optional
		Whether to output the frames, and if so, how should it be shown.

		The options are:

		- :func:`DisplayConfig.Headless`: No display.
		- :func:`DisplayConfig.OpenCV`: Native OpenCV display option, creates a window using Qt.
		- :func:`DisplayConfig.Jupyter`: Jupyter Notebook display using widgets.

		If no value or `None` is passed, defaults :func:`DisplayConfig.Headless`.
	"""

	# === Input sanitisation ===

	if isinstance(capture_location, int): # Webcam input
		if capture_location < 0:
			raise ValueError('Integer input must be a positive number.')
	else: # Filepath input
		# Convert string to Path object for easier operation.
		if isinstance(capture_location, str):
			capture_location = Path(capture_location)

		if not capture_location.exists():
			raise FileNotFoundError('No file was found at this location.')

	vid_cap = cv.VideoCapture(capture_location)

	if not vid_cap.isOpened():
		print('Cannot open video source.')
		return

	# Null sentinel
	if display_config is None:
		display_config = DisplayConfig.Headless()

	if isinstance(display_config, DisplayConfig.Jupyter):
		display(display_config.image_widget)

	try:
		while _frame_loop(
			vid_cap=vid_cap,
			callback=callback,
			display_option=display_config
		): pass

	except KeyboardInterrupt:
		print('Video stream interrupted.')

	finally:
		vid_cap.release()

		match display_config:
			case DisplayConfig.OpenCV():
				cv.destroyAllWindows()
			case _:
				pass
