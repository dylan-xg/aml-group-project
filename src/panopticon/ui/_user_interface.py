"""The main window."""

import tkinter as _tk
from tkinter import ttk as _ttk
from typing import Iterable as _Iterable

from ..recognition import BaseModel
from ..settings import SETTINGS
from ._video_widget import VideoWidget
from ._button import Button
from ..video_feed import VideoFeed


class UserInterface:
	def __init__(
		self,
		width: int = SETTINGS.WINDOW_WIDTH,
		height: int = SETTINGS.WINDOW_HEIGHT,
		title: str | None = None,
		video_feed: VideoFeed | None = None
	) -> None:

		if width < 1 or height < 1: raise ValueError('Window cannot have a side length of 0 or less')

		self.window = _tk.Tk()

		if title is not None: self.window.title(string=title)

		resolution: str = ''.join([
			str(object=width),
			'x',
			str(object=height)
		])
		self.window.geometry(newGeometry=resolution)

		self.mainframe = _ttk.Frame(master=self.window, padding=(0,0,0,0))
		self.mainframe.pack(expand=True, fill=_tk.BOTH)

		# Required for resizing to work.
		self.mainframe.rowconfigure(index=0, weight=1)
		self.mainframe.columnconfigure(index=0, weight=1)
		# Input panel is fixed width.
		self.mainframe.columnconfigure(index=1, weight=0)

		# Create a dedicated container frame for the video.
		self.video_container = _ttk.Frame(master=self.mainframe, padding=(0,0,0,0))
		self.video_container.grid(row=0, column=0, sticky='news')
		# This is required to allow the container to dictate the video size.
		self.video_container.pack_propagate(flag=False)

		# Create a side panel for input buttons.
		self.input_panel = _ttk.Frame(master=self.mainframe, padding=(0,0,0,0))
		self.input_panel.grid(row=0, column=1, sticky='news')

		if video_feed is not None:
			self.add_feed(video_feed, auto_start=True)


	def add_feed(
		self,
		video_feed: VideoFeed,
		/,
		auto_start: bool = False
	) -> None:
		if getattr(self, 'video_widget', False): raise ValueError('Feed already added.')
		self.video_widget = VideoWidget(
			parent_frame=self.video_container,
			video_feed=video_feed
		)
		self.restart_button: Button = Button.simple_button(
			input_panel=self.input_panel,
			label='Restart Video',
			order=20,
			command=self.video_widget.restart_video
		)
		if auto_start: self.video_widget.start_playback()


	def add_models(
		self,
		models: _Iterable[BaseModel],
		order: int = 100
	) -> None:
		"""Add support for a collection of models."""
		# Need to know how each model will be structured.

		model_buttons: list[Button] = []

		for i, model in enumerate(models):
			new_button: Button = Button.complex_button(
				input_panel=self.input_panel,
				label=model.name,
				order=order + i,
				command=model.toggle_enabled
			)
			model_buttons.append(new_button)
			# Only important thing to extract is the inference function.


	def start(self) -> None:
		self.window.mainloop()
