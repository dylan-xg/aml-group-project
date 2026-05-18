"""A tkinter widget that displays the output video."""

import tkinter as _tk
from tkinter import ttk as _ttk
from PIL.Image import fromarray as _np2pil
from PIL.ImageTk import PhotoImage as _Image
import cv2 as _cv

from ..typing import Frame
from ..video_processor import VideoFeed


class VideoWidget:
	def __init__(
		self,
		parent_frame: _ttk.Frame,
		video_feed: VideoFeed
	) -> None:
		self.parent: _ttk.Frame = parent_frame

		# Access the root window through the parent frame.
		top_level: _tk.Tk | _tk.Toplevel = self.parent.winfo_toplevel()
		if isinstance(top_level, _tk.Toplevel): raise ValueError
		self.root: _tk.Tk = top_level

		self.video_feed: VideoFeed = video_feed

		# Initialise with the requested width and height of the parent frame.
		self.target_width: int = self.parent.winfo_width()
		self.target_height: int = self.parent.winfo_height()

		self.label_widget = _ttk.Label(master=self.parent)
		# Expand the label to fill the parent frame.
		self.label_widget.pack(expand=True, fill=_tk.BOTH)

		self.current_image: _Image | None = None

		# Bind the configure event to detect when the frame changes size
		self.parent.bind(sequence='<Configure>', func=self.on_resize)


	def on_resize(self, event) -> None:
		# Update target dimensions based on the new size of the parent frame.
		# We ensure dimensions do not drop below 1 to prevent _cv.resize errors.
		self.target_width = max(1, event.width)
		self.target_height = max(1, event.height)


	def _resize_frame(self, frame: Frame) -> Frame:
		"""Resize the provided frame to the available space."""
		# Retrieve original video dimensions
		orig_height: int
		orig_width: int
		orig_height, orig_width = frame.shape[:2]

		# Calculate scaling factor to fit within the target dimensions
		scale_width: float = self.target_width / orig_width
		scale_height: float = self.target_height / orig_height
		scale: float = min(scale_width, scale_height) # Use min() to ensure it fits without cropping

		new_width: int = max(1, int(orig_width * scale))
		new_height: int = max(1, int(orig_height * scale))

		if new_width < 1 or new_height < 1:
			raise ValueError('Cannot resize to a side length of 0 or less')

		# Resize proportionally
		return _cv.resize(src=frame, dsize=(new_width, new_height))


	def update_frame(self) -> None:
		try:
			frame: Frame = self.video_feed.process_frame()
		except ValueError:
			return

		frame = self._resize_frame(frame=frame)

		self.current_image = _Image(image=_np2pil(obj=frame))
		self.label_widget.configure(image=self.current_image)

		FRAMETIME_MS: int = int(self.video_feed.frametime * 1000)
		self.root.after(ms=FRAMETIME_MS, func=self.update_frame)
