"""The main window."""

import tkinter as _tk
from tkinter import ttk as _ttk
from typing import Iterable as _Iterable

from panopticon.modules import BaseModule as _BaseModule
from panopticon.recognition import FaceRecognitionSystem as _FaceRecognitionSystem
from panopticon.settings import SETTINGS as _SETTINGS
from panopticon.video_feed import VideoFeed as VideoFeed

from ._button import Button as _Button
from ._video_widget import VideoWidget as _VideoWidget


class UserInterface:
	def __init__(
		self,
		width: int = _SETTINGS.WINDOW_WIDTH,
		height: int = _SETTINGS.WINDOW_HEIGHT,
		title: str | None = None,
		video_feed: VideoFeed | None = None,
	) -> None:

		if width < 1 or height < 1:
			raise ValueError("Window cannot have a side length of 0 or less")

		self.window = _tk.Tk()

		if title is not None:
			self.window.title(string=title)

		self.window.geometry(newGeometry=f"{width}x{height}")

		# MAIN FRAME
		self.mainframe = _ttk.Frame(master=self.window)
		self.mainframe.pack(expand=True, fill=_tk.BOTH)

		self.mainframe.rowconfigure(index=0, weight=1)
		self.mainframe.columnconfigure(index=0, weight=1)
		self.mainframe.columnconfigure(index=1, weight=0)

		# VIDEO
		self.video_container = _ttk.Frame(master=self.mainframe)
		self.video_container.grid(row=0, column=0, sticky="news")
		self.video_container.pack_propagate(flag=False)

		# INPUT PANEL
		self.input_panel = _ttk.Frame(master=self.mainframe, padding=10)
		self.input_panel.grid(row=0, column=1, sticky="ns")

		# TITLE
		self.title_label = _ttk.Label(
			master=self.input_panel,
			text="FACE RECOGNITION SYSTEM",
			font=("Arial", 14, "bold"),
		)
		self.title_label.grid(row=0, column=0, pady=10)

		# SECTIONS
		self.module_label = _ttk.Label(
			master=self.input_panel, text="MODULES", font=("Arial", 11, "bold")
		)
		self.module_label.grid(row=10, column=0, pady=(20, 5), sticky="w")

		self.registration_label = _ttk.Label(
			master=self.input_panel, text="REGISTRATION", font=("Arial", 11, "bold")
		)
		self.registration_label.grid(row=30, column=0, pady=(20, 5), sticky="w")

		self.register_button: _Button = _Button.simple_button(
			input_panel=self.input_panel,
			label="Register new face",
			order=35,
			command=self.open_registration_popup,
		)

		# VIDEO FEED
		if video_feed is not None:
			self.add_feed(video_feed=video_feed, auto_start=True)

	# VIDEO FEED

	def add_feed(self, video_feed: VideoFeed, auto_start: bool = False) -> None:

		if hasattr(self, "video_widget"):
			raise ValueError("Feed already added.")

		self.video_widget: _VideoWidget = _VideoWidget(
			parent_frame=self.video_container, video_feed=video_feed
		)

		self.restart_button: _Button = _Button.simple_button(
			input_panel=self.input_panel,
			label="Restart Video",
			order=5,
			command=self.video_widget.restart_video,
		)

		if auto_start:
			self.video_widget.start_playback()

	# MODULE BUTTONS

	def add_modules(self, models: _Iterable[_BaseModule], order: int = 11) -> None:

		for i, module in enumerate(iterable=models):
			_Button.complex_button(
				input_panel=self.input_panel,
				label=module.name,
				order=order + i,
				command=module.toggle_enabled,
			)

	# REGISTRATION POPUP

	def open_registration_popup(self) -> None:

		popup = _tk.Toplevel(master=self.window)
		popup.title(string="Register New Face")
		popup.geometry(newGeometry="350x220")

		title = _tk.Label(
			master=popup, text="Face Registration", font=("Arial", 16, "bold")
		)
		title.pack(pady=15)

		name_label = _tk.Label(master=popup, text="Enter Name:")
		name_label.pack()

		name_entry = _tk.Entry(master=popup, width=30)
		name_entry.pack(pady=10)

		confirm_button = _tk.Button(
			master=popup,
			text="Register",
			command=lambda: self.confirm_registration(
				name=name_entry.get(), popup=popup
			),
		)
		confirm_button.pack(pady=20)

	# CONFIRM REGISTRATION

	def confirm_registration(self, name: str, popup: _tk.Toplevel) -> None:

		if name.strip() == "":
			return

		print(f"Registering {name}...")
		_FaceRecognitionSystem.enable_registration_mode(name.strip())
		popup.destroy()

	# START UI

	def start(self) -> None:
		self.window.mainloop()
