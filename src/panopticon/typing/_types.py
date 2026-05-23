
from collections.abc import Callable as _Callable
from typing import Any as _Any

import cv2 as _cv


type Frame = _cv.typing.MatLike
"""A frame is a single image from a webcam or video input."""

type ProcessFrameCallback = _Callable[[Frame], Frame | None]
"""A function that is called on a frame input for processing of some kind. Can return a modified result."""

type FrameDisplayCallback = _Callable[[Frame], None]
"""A function used for a custom frame display output."""
