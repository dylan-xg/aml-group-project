from typing import TypeAlias as _TypeAlias

import cv2 as _cv

#__all__: list[str] = [x for x in dir() if not x.startswith('_')]
#__all__: list[str] = ['Frame']

Frame: _TypeAlias = _cv.typing.MatLike
