"""The main window."""

import tkinter as _tk
from tkinter import ttk as _ttk
from typing import Iterable as _Iterable

from ..modules import BaseModule
from ..settings import SETTINGS
from ..video_feed import VideoFeed
from ._button import Button
from ._video_widget import VideoWidget

from panopticon.recognition import _run


class UserInterface:
	def __init__(
		self,
		width: int = SETTINGS.WINDOW_WIDTH,
		height: int = SETTINGS.WINDOW_HEIGHT,
		title: str | None = None,
		video_feed: VideoFeed | None = None,
	) -> None:

		if width < 1 or height < 1:
			raise ValueError("Window cannot have a side length of 0 or less")

		self.window = _tk.Tk()

		if title is not None:
			self.window.title(title)

		self.window.geometry(f"{width}x{height}")

		# MAIN FRAME
		self.mainframe = _ttk.Frame(self.window)
		self.mainframe.pack(expand=True, fill=_tk.BOTH)

		self.mainframe.rowconfigure(0, weight=1)
		self.mainframe.columnconfigure(0, weight=1)
		self.mainframe.columnconfigure(1, weight=0)

		# VIDEO
		self.video_container = _ttk.Frame(self.mainframe)
		self.video_container.grid(row=0, column=0, sticky="news")
		self.video_container.pack_propagate(False)

		# INPUT PANEL
		self.input_panel = _ttk.Frame(self.mainframe, padding=10)
		self.input_panel.grid(row=0, column=1, sticky="ns")

		# TITLE
		self.title_label = _ttk.Label(
			self.input_panel, text="FACE RECOGNITION SYSTEM", font=("Arial", 14, "bold")
		)
		self.title_label.grid(row=0, column=0, pady=10)

		# SECTIONS
		self.module_label = _ttk.Label(
			self.input_panel, text="MODULES", font=("Arial", 11, "bold")
		)
		self.module_label.grid(row=10, column=0, pady=(20, 5), sticky="w")

		self.registration_label = _ttk.Label(
			self.input_panel, text="REGISTRATION", font=("Arial", 11, "bold")
		)
		self.registration_label.grid(row=30, column=0, pady=(20, 5), sticky="w")

		self.register_button = Button.simple_button(
			self.input_panel, "Register new face", 35, self.open_registration_popup
		)

		# VIDEO FEED
		if video_feed is not None:
			self.add_feed(video_feed, auto_start=True)

	# VIDEO FEED

	def add_feed(self, video_feed: VideoFeed, auto_start: bool = False) -> None:

		if hasattr(self, "video_widget"):
			raise ValueError("Feed already added.")

		self.video_widget = VideoWidget(
			parent_frame=self.video_container, video_feed=video_feed
		)

		self.restart_button = Button.simple_button(
			input_panel=self.input_panel,
			label="Restart Video",
			order=5,
			command=self.video_widget.restart_video,
		)

		if auto_start:
			self.video_widget.start_playback()

	# MODULE BUTTONS

	def add_modules(self, models: _Iterable[BaseModule], order: int = 11) -> None:

		for i, module in enumerate(models):
			Button.complex_button(
				input_panel=self.input_panel,
				label=module.name,
				order=order + i,
				command=module.toggle_enabled,
			)

	# REGISTRATION POPUP

	def open_registration_popup(self):

		popup = _tk.Toplevel(self.window)
		popup.title("Register New Face")
		popup.geometry("350x220")

		title = _tk.Label(popup, text="Face Registration", font=("Arial", 16, "bold"))
		title.pack(pady=15)

		name_label = _tk.Label(popup, text="Enter Name:")
		name_label.pack()

		name_entry = _tk.Entry(popup, width=30)
		name_entry.pack(pady=10)

		confirm_button = _tk.Button(
			popup,
			text="Register",
			command=lambda: self.confirm_registration(name_entry.get(), popup),
		)
		confirm_button.pack(pady=20)

	# CONFIRM REGISTRATION

	def confirm_registration(self, name: str, popup):

		if name.strip() == "":
			return

		print(f"Registering {name}...")
		_run.REGISTRATION_MODE = True
		_run.REGISTRATION_NAME = name.strip()
		_run.REGISTRAION_FRAMES = []

		popup.destroy()

	# START UI

	def start(self) -> None:
		self.window.mainloop()
