"""The main window."""

import tkinter as _tk
from tkinter import ttk as _ttk
from typing import Callable as _Callable

from ._video_widget import VideoWidget
from ..video_feed import VideoFeed


class UserInterface:
	def __init__(
		self,
		width: int,
		height: int,
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
			self.add_feed(video_feed)


	def add_button(self, label: str, order: int, command: _Callable) -> _ttk.Button:
		new_button = _ttk.Button(
			master=self.input_panel,
			text=label,
			command=command
		)
		new_button.grid(
			row=order,
			column=0,
			sticky='news',
			padx=5,
			pady=5
		)
		self.input_panel.rowconfigure(index=order, weight=0)
		return new_button


	def add_feed(self, video_feed: VideoFeed, /) -> None:
		if getattr(self, 'video_widget', False): raise ValueError('Feed already added.')
		self.video_widget = VideoWidget(self.video_container, video_feed)
		#self.start_button = self.add_button(
		#	label='Start Video',
		#	order=10,
		#	command=self.video_widget.start_playback
		#)
		self.restart_button = self.add_button(
			label='Restart Video',
			order=20,
			command=self.video_widget.restart_video
		)


	def start(self) -> None:
		self.video_widget.start_playback()
		self.window.mainloop()
