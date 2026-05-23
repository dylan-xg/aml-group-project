
from tkinter import ttk as _ttk

from ..typing import (
	ButtonCommand,
	ModelStateCallback, # Needed for docstring.
	ButtonCommandWithStateCallback
)
from ..settings import SETTINGS


class Button:
	"""A wrapper for the tkinter button to provide extra functionality and control.

	Should only be created using the class factory methods :func:`simple_button` and :func:`complex_button`.
	"""

	@property
	def label(self) -> str:
		return self._label

	@label.setter
	def label(self, val: str) -> None:
		self._label = val


	@property
	def order(self) -> int:
		return self._order

	@order.setter
	def order(self, val: int) -> None:
		self._order = val


	def __init__(
		self,
		label: str,
		order: int,
		button: _ttk.Button
	) -> None:
		"""Should not be called directly."""
		self._label: str = label
		self._order: int = order
		self._button: _ttk.Button = button
		self._button.grid(
			row=order,
			column=0,
			sticky='news',
			padx=5,
			pady=5
		)

	@classmethod
	def simple_button(
		cls,
		input_panel: _ttk.Frame,
		label: str,
		order: int,
		command: ButtonCommand
	):
		"""Class factory method to build a button with a simple button command."""
		inst = cls(
			label,
			order,
			_ttk.Button(
				master=input_panel,
				text=label,
				command=command,
				width=SETTINGS.INPUT_BUTTON_WIDTH
			)
		)
		input_panel.rowconfigure(index=order, weight=0)
		return inst

	@classmethod
	def complex_button(
		cls,
		input_panel: _ttk.Frame,
		label: str,
		order: int,
		command: ButtonCommandWithStateCallback
	):
		"""Class factory method to build a button with a command that supports :func:`ModelStateCallback`."""
		inst = cls(
			label,
			order,
			_ttk.Button(
				master=input_panel,
				text=f'{label} (Disabled)',
				command=lambda:command(inst.update_label),
				width=SETTINGS.INPUT_BUTTON_WIDTH
			)
		)
		input_panel.rowconfigure(index=order, weight=0)
		return inst


	def update_label(self, state: bool) -> None:
		state_label = 'Enabled' if state else 'Disabled'
		self._button.configure(text=f'{self._label} ({state_label})')
