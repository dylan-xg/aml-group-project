"""This module handles video input and video display.

Example
-------
>>> from panopticon import video_processor as VP
>>> WEBCAM_SOURCE = 0
>>> def detect_face(frame): ...
>>> VP.process_video(
... 	capture_location=WEBCAM_SOURCE,
... 	callback=detect_face,
... 	display_config=VP.DisplayConfig.OpenCV(frametime=1/25)
... )
"""

import time
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from IPython.display import display

from . import display_config as DisplayConfig
from .display_config import DisplayConfigType as ConfigType
from .settings import Frame, ProcessFrameCallback


_END_LOOP = False
_GO_AGAIN = True


def _display_video(display_option: ConfigType, frame: Frame, verbosity: int) -> bool:
	"""Will attempt to display an output based on the display option.

	Private function, should not be called directly.
	"""

	match display_option:
		case DisplayConfig.OpenCV():
			cv2.imshow(winname="frame", mat=frame)

		case DisplayConfig.Jupyter(image_widget=widget):
			ret: bool
			buffer: npt.NDArray[np.uint8]
			ret, buffer = cv2.imencode(ext=".jpg", img=frame)

			if not ret:
				if verbosity > 0:
					print("Can't encode frame as image. Exiting ...")
				return _END_LOOP

			widget.value = buffer.tobytes()

		case DisplayConfig.Custom(func=display_func):
			display_func(frame)

		case _:  # DisplayConfig.Headless:
			pass

	return _GO_AGAIN


def _frame_loop(
	vid_cap: cv2.VideoCapture,
	callback: ProcessFrameCallback | None,
	display_option: ConfigType,
	verbosity: int,
) -> bool:
	"""This function is called continuously until either the video ends, or is interrupted.

	Private function, should not be called directly.
	"""

	starttime: float = time.time()

	ret: bool
	frame: Frame
	ret, frame = vid_cap.read()

	if not ret:
		if verbosity > 0:
			print("Can't receive frame (stream end?). Exiting ...")
		return _END_LOOP

	frame_result: Frame | None = callback(frame) if callback is not None else None
	if frame_result is None:
		frame_result = frame

	if not _display_video(
		display_option=display_option, frame=frame_result, verbosity=verbosity
	):
		return _END_LOOP

	def _wait_for(length: float, /) -> None:
		elapsedtime: float = time.time() - starttime
		time_to_wait: float = length - elapsedtime
		if time_to_wait > 0:
			time.sleep(time_to_wait)

	match display_option:
		case DisplayConfig.OpenCV(frametime=ft):
			# Frametime is in seconds, waitKey expects milliseconds
			if cv2.waitKey(delay=int(ft * 1000)) == ord("q"):
				return _END_LOOP

		case DisplayConfig.Jupyter(frametime=ft):
			_wait_for(ft)

		case DisplayConfig.Custom(frametime=ft):
			_wait_for(ft)

		case _:  # DisplayConfig.Headless:
			pass

	return _GO_AGAIN


def process_video(
	capture_location: int | str | Path,
	callback: ProcessFrameCallback | None = None,
	display_config: ConfigType | None = None,
	*,
	verbosity: int = 0,
) -> None:
	"""Read from a video input and apply the callback to it, then optionally display it.

	Parameters
	----------
	capture_location : int | str | Path
		What the video source is.

		- For a webcam input, use an `int`, 0 is the default webcam.
		- For a video file, pass in a file location either as `str` or `Path` (preferred).

	callback : :func:`ProcessFrameCallback`, optional
		A function that will be called every frame with every :func:`Frame`.

		This function can return a :func:`Frame` which will be used for display if enabled.

		:func:`ProcessFrameCallback` = `(Frame) -> Frame | None`.

		:func:`Frame` = `cv.typing.MatLike`

	display_config : :func:`DisplayConfigType`, optional
		Whether to output the frames, and if so, how should it be shown.

		The options are:

		- :func:`DisplayConfig.Headless`: No display.
		- :func:`DisplayConfig.OpenCV`: Native OpenCV display option, creates a window using Qt.
		- :func:`DisplayConfig.Jupyter`: Jupyter Notebook display using widgets.

		If no value or `None` is passed, defaults :func:`DisplayConfig.Headless`.

	verbosity : int, optional
		Scales how much information should be output. Used for debugging.
	"""

	# === Input sanitisation ===

	if isinstance(capture_location, int):  # Webcam input
		if capture_location < 0:
			raise ValueError("Integer input must be a positive number.")
	else:  # Filepath input
		# Convert string to Path object for easier operation.
		if isinstance(capture_location, str):
			capture_location = Path(capture_location)

		if not capture_location.exists():
			raise FileNotFoundError("No file was found at this location.")

	vid_cap = cv2.VideoCapture(capture_location)

	if not vid_cap.isOpened():
		if verbosity > 0:
			print("Cannot open video source.")
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
			display_option=display_config,
			verbosity=verbosity,
		):
			pass

	except KeyboardInterrupt:
		if verbosity > 0:
			print("Video stream interrupted.")

	finally:
		vid_cap.release()

		match display_config:
			case DisplayConfig.OpenCV():
				cv2.destroyAllWindows()
			case _:
				pass
