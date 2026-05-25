
from dataclasses import (
	dataclass as _dataclass,
	astuple as _astuple
)

import cv2 as _cv

from src.panopticon.typing import Frame


type COLOUR = tuple[int, int, int]

BOX_COLOUR = (0, 0, 255)
"""BGR format."""
TEXT_COLOUR = (0, 0, 0)
"""BGR format."""
FONT = _cv.FONT_HERSHEY_SIMPLEX


@_dataclass
class Box:
	left: int
	top: int
	right: int
	bottom: int
	colour: COLOUR = BOX_COLOUR

	def draw_onto_frame(self, frame: Frame, /) -> Frame:
		return _cv.rectangle(
			img=frame,
			rec=(self.left, self.top, self.right, self.bottom),
			color=self.colour
		)


@_dataclass
class Text:
	label: str
	position: tuple[int, int]
	"""(left, bottom)"""
	scale: float = 1
	colour: COLOUR = TEXT_COLOUR
	thickness: int = 2

	#_cv.getFontScaleFromHeight
	#_cv.getTextSize

	def draw_onto_frame(self, frame: Frame, /) -> Frame:
		return _cv.putText(
			img=frame,
			text=self.label,
			org=self.position,
			fontFace=FONT,
			fontScale=self.scale,
			color=self.colour,
			thickness=self.thickness,
			lineType=_cv.LINE_AA,
		)


@_dataclass
class Face:
	box: Box
	details: list[Text]

	def draw_onto_frame(self, frame: Frame, /) -> Frame:
		frame = self.box.draw_onto_frame(frame)

		for detail in self.details:
			frame = detail.draw_onto_frame(frame)

		return frame


class Drawer:
	_faces: list[Face] = []
	_texts: list[Text] = []

	def add_face(self, face: Face | list[Face], /) -> None:
		if isinstance(face, Face): face = [face]
		if face in self._faces: raise ValueError('Face already exists')
		self._faces += face

	def add_text(self, text: Text | list[Text], /) -> None:
		if isinstance(text, Text): text = [text]
		if text in self._texts: raise ValueError('Face already exists')
		self._texts += text


	def draw_onto_frame(self, frame: Frame, /) -> Frame:
		for face in self._faces:
			frame = face.draw_onto_frame(frame)

		for text in self._texts:
			frame = text.draw_onto_frame(frame)

		return frame
