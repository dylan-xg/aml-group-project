"""A tkinter widget that displays the output video."""

import tkinter as _tk
from tkinter import ttk as _ttk
from pathlib import Path as _Path
from PIL.Image import fromarray as _np2pil
from PIL.ImageTk import PhotoImage as _Image
import cv2 as _cv

from ..typing import Frame

class VideoWidget:
	def __init__(
		self,
		parent_frame: _ttk.Frame,
		test_path: _Path
	) -> None:
		self.parent: _ttk.Frame = parent_frame

		# Access the root window through the parent frame.
		self.root = self.parent.winfo_toplevel()

		if not test_path.exists():
			raise FileNotFoundError('No file was found at this location.')

		self.vid = _cv.VideoCapture(test_path)

		if not self.vid.isOpened():
			print(f'Failed to open video file: {test_path}')
			self.root.quit()
			return

		# Initialise with the requested width and height of the parent frame.

		self.target_width: int = self.parent.winfo_width()
		self.target_height: int = self.parent.winfo_height()
		#self.target_width: int = self.parent.cget('width') or 800
		#self.target_height: int = self.parent.cget('height') or 600

		self.label_widget = _ttk.Label(self.parent)
		# Expand the label to fill the parent frame.
		self.label_widget.pack(expand=True, fill=_tk.BOTH)

		self.current_image = None

		# Bind the configure event to detect when the frame changes size
		self.parent.bind('<Configure>', self.on_resize)


	def on_resize(self, event):
		# Update target dimensions based on the new size of the parent frame.
		# We ensure dimensions do not drop below 1 to prevent _cv.resize errors.
		self.target_width = max(1, event.width)
		self.target_height = max(1, event.height)


	def _resize_frame(self, frame: Frame) -> Frame:
		# Retrieve original video dimensions
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
		return _cv.resize(frame, (new_width, new_height))


	def update_frame(self):
		ret: bool
		frame: Frame
		ret, frame = self.vid.read()

		if not ret: return

		frame: Frame = self._resize_frame(frame)

		opencv_image = _cv.cvtColor(frame, _cv.COLOR_BGR2RGBA)

		self.current_image = _Image(image=_np2pil(opencv_image))
		self.label_widget.configure(image=self.current_image)

		FRAMETIME_MS = 1000 // 60
		self.root.after(FRAMETIME_MS, self.update_frame)
