import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2 as cv
import numpy as np

import ipywidgets as widgets
from IPython.display import display


# This probably isn't needed since everything is in its own module now.
class VideoProcessor:

	@dataclass
	class DisplayConfig:
		pass

	@dataclass
	class DisplayOption_Headless(DisplayConfig):
		pass

	@dataclass
	class DisplayOption_Jupyter(DisplayConfig):
		image_widget: widgets.Image

	@dataclass
	class DisplayOption_OpenCV(DisplayConfig):
		pass

	#DisplayConfig: TypeAlias = DisplayOption_Headless | DisplayOption_Jupyter | DisplayOption_OpenCV

	FrameCallbackType: TypeAlias = Callable[[cv.typing.MatLike], cv.typing.MatLike | None]

	_END_LOOP = False
	_GO_AGAIN = True

	@staticmethod
	def _display_video(
		display_option: DisplayConfig,
		frame: cv.typing.MatLike
	) -> bool:
		"""Will attempt to display an output based on the display option."""

		match display_option:
			case VideoProcessor.DisplayOption_Jupyter(image_widget=image_widget):
				ret: bool
				buffer: np.ndarray[Any, np.dtype[np.uint8]]
				ret, buffer = cv.imencode(ext='.jpg', img=frame)

				if not ret:
					print("Can't encode frame as image. Exiting ...")
					return VideoProcessor._END_LOOP

				image_widget.value = buffer.tobytes()

			case VideoProcessor.DisplayOption_OpenCV:
				print('Frame should be displayed')
				cv.imshow(winname='frame', mat=frame)

		return VideoProcessor._GO_AGAIN

	@staticmethod
	def _frame_loop(
		vid_cap: cv.VideoCapture,
		callback: FrameCallbackType,
		display_option: DisplayConfig,
		frametime: float
	) -> bool:
		"""This function is called continuously until either the video ends, or is interrupted."""

		start_time: float = time.time()

		ret: bool
		frame: cv.typing.MatLike
		ret, frame = vid_cap.read()

		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			return VideoProcessor._END_LOOP

		frame_result: cv.typing.MatLike | None = callback(frame)
		if frame_result is None: frame_result = frame

		if not VideoProcessor._display_video(
			display_option=display_option,
			frame=frame_result
		): return VideoProcessor._END_LOOP

		match display_option:
			case VideoProcessor.DisplayOption_OpenCV:
				if cv.waitKey(delay=int(1/frametime)) == ord('q'):
					return VideoProcessor._END_LOOP

			case _: # VideoProcessor.DisplayOption_Jupyter:
				# Enforce the framerate pacing
				elapsed_time: float = time.time() - start_time
				time_to_wait: float = frametime - elapsed_time
				if time_to_wait > 0: time.sleep(time_to_wait)

		return VideoProcessor._GO_AGAIN

	@staticmethod
	def process_video(
		capture_location: int | str | Path,
		callback: FrameCallbackType,
		*,
		display_config: DisplayConfig | None = None,
		frametime: float = 1.0 / 30.0
	) -> None:
		"""Read from a video input and apply the callback to it.

		Args:
			capture_location (int | str | Path): What the video source is.
				- For a webcam input, use an int.
				- For a video file, use a str or Path.
			callback (FrameCallbackType: Callable[[cv.typing.MatLike], cv.typing.MatLike | None]): Function that will be called with each frame.
			frametime (float): In seconds, how long between each frame. Use 1 / fps if you want to pass in a framerate.
			display_type (DisplayType): Whether to output the frames, and if so, how should it be shown. Options:
				- Jupyter Notebook.
				- Qt window via OpenCV.

		Returns:
			None:
		"""

		# Null sentinel
		if display_config is None:
			display_config = VideoProcessor.DisplayOption_Jupyter(
				# JPEG is faster than PNG
				image_widget=widgets.Image(format='jpeg')
			)

		vid_cap = cv.VideoCapture(capture_location)

		if not vid_cap.isOpened():
			print('Cannot open video source.')
			return

		if isinstance(display_config, VideoProcessor.DisplayOption_Jupyter):
			display(display_config.image_widget)

		try:
			while VideoProcessor._frame_loop(
				vid_cap=vid_cap,
				callback=callback,
				display_option=display_config,
				frametime=frametime
			): pass

		except KeyboardInterrupt:
			print('Video stream interrupted.')

		finally:
			vid_cap.release()
