"""The main window."""

import tkinter as _tk
from tkinter import ttk as _ttk

import cv2 as cv

from ._video_widget import VideoWidget

class UserInterface:
	def __init__(
		self,
		width: int,
		height: int,
		title: str | None = None,
	) -> None:

		self.window = _tk.Tk()

		if title is not None: self.window.title(string=title)

		resolution: str = ''.join([
			str(object=width),
			'x',
			str(object=height)
		])
		self.window.geometry(newGeometry=resolution)

		self.mainframe = _ttk.Frame(master=self.window)
		self.mainframe.pack(expand=True, fill=_tk.BOTH)

		self.mainframe.rowconfigure(index=0, weight=1)
		self.mainframe.columnconfigure(index=0, weight=8)
		self.mainframe.columnconfigure(index=1, weight=1)

		# Create a dedicated container frame for the video.
		self.video_container = _ttk.Frame(master=self.mainframe)
		self.video_container.grid(row=0, column=0, sticky='news')

		self.video_container.pack_propagate(flag=False)

		# Create a side panel with buttons.
		self.input_panel = _ttk.Frame(master=self.mainframe)
		self.input_panel.grid(row=0, column=1, sticky='news')

		self.input_panel.rowconfigure(index=0, weight=0)
		self.input_panel.rowconfigure(index=1, weight=0)
		self.input_panel.columnconfigure(index=0, weight=1)

		self.start_button = _ttk.Button(
			master=self.input_panel,
			text='Start Video'
		)
		self.start_button.grid(
			row=0,
			column=0,
			sticky='news',
			padx=5,
			pady=5
		)

		self.restart_button = _ttk.Button(
			master=self.input_panel,
			text='Restart Video'
		)
		self.restart_button.grid(
			row=1,
			column=0,
			sticky='news',
			padx=5,
			pady=5
		)


	def add_feed(self, video_widget: VideoWidget, /) -> None:
		self.start_button.configure(command=video_widget.update_frame)
		self.restart_button.configure(
			command=lambda: video_widget.video_feed.vid_cap.set(
				propId=cv.CAP_PROP_POS_FRAMES,
				value=0
			)
		)


	def start(self) -> None:
		self.window.mainloop()
