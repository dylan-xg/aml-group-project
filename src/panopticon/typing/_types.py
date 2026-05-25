
from collections.abc import Callable as _Callable
from typing import Any as _Any

import cv2 as _cv


type Frame = _cv.typing.MatLike
"""A frame is a single image from a webcam or video input."""

type ProcessFrameCallback = _Callable[[Frame], Frame | None]
"""A function that is called on a frame input for processing of some kind. Can return a modified result."""

type ButtonCommand = _Callable[[], _Any]
"""The function signature of the command called when a button is pressed."""

type ModelStateCallback = _Callable[[bool], None]
"""A callback function to report the current state of the model."""

type ButtonCommandWithStateCallback = _Callable[[ModelStateCallback], None]
"""An expanded button command that has an input for the :func:`ModelStateCallback` function."""
