# This file turns the containing folder into a sub-package.
"""This is the tkinter interface."""

from ._user_interface import UserInterface as UserInterface

from ._video_widget import VideoWidget as VideoWidget

__all__: list[str] = ['UserInterface', 'VideoWidget']
