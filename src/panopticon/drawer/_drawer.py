"""Provides dataclasses and methods for drawing onto a frame."""

from dataclasses import dataclass as _dataclass

import cv2 as _cv

from src.panopticon.typing import (
	Colour as _Colour,
	Frame as _Frame,
	Position as _Position,
)


COLOUR_DEFAULT_BOX: _Colour = (0, 0, 255)  # BGR format
COLOUR_DEFAULT_TEXT: _Colour = (0, 0, 0)  # BGR format
FONT = _cv.FONT_HERSHEY_SIMPLEX
PADDING_DEFAULT_VERTICAL = 5
PADDING_DEFAULT_HORIZONTAL = 5


@_dataclass
class Box:
	left: int
	top: int
	right: int
	bottom: int
	colour: _Colour = COLOUR_DEFAULT_BOX

	def draw_onto_frame(self, frame: _Frame, /) -> _Frame:
		return _cv.rectangle(
			img=frame,
			pt1=(self.left, self.top),
			pt2=(self.right, self.bottom),
			color=self.colour,
		)


@_dataclass
class Text:
	label: str
	scale: float = 1
	colour: _Colour = COLOUR_DEFAULT_TEXT
	thickness: int = 2
	position: _Position = (0, 0)
	"""Text position (left, bottom).

	Will override the position passed into :func:`draw_onto_frame`.
	"""

	def draw_onto_frame(self, frame: _Frame, /, pos: _Position | None = None) -> _Frame:
		return _cv.putText(
			img=frame,
			text=self.label,
			org=pos or self.position,
			fontFace=FONT,
			fontScale=self.scale,
			color=self.colour,
			thickness=self.thickness,
			lineType=_cv.LINE_AA,
		)


class Face:
	"""A class that represents one face in the frame.

	Parameters
	----------
	image : frame
		A cropped section containing just the face.
	box : box
		Denotes the position of the face in the frame.
	texts : list of text, optional
		Prefilled information displayed under the face.
	padding_box_y : int, optional
		The vertical padding around the face box.
	padding_frame_x : int, optional
		The horizontal padding around the edge of the frame.
	padding_frame_y : int, optional
		The vertical padding around the edge of the frame.
	padding_text_y : int, optional
		The vertical padding around the text.
	"""

	def __init__(
		self,
		image: _Frame,
		box: Box,
		texts: list[Text] = [],
		padding_box_y: int = PADDING_DEFAULT_VERTICAL,
		padding_frame_x: int = PADDING_DEFAULT_HORIZONTAL,
		padding_frame_y: int = PADDING_DEFAULT_VERTICAL,
		padding_text_y: int = PADDING_DEFAULT_VERTICAL,
	) -> None:
		self._image = image
		self._box = box
		self._texts = texts

		##self.padding_box_x: int = PADDING_DEFAULT_HORIZONTAL # Unused
		self.padding_box_y: int = padding_box_y
		self.padding_frame_x: int = padding_frame_x
		self.padding_frame_y: int = padding_frame_y
		self.padding_text_y: int = padding_text_y

	@property
	def image(self) -> _Frame:
		return self._image

	@property
	def box(self) -> Box:
		return self._box

	@property
	def texts(self) -> list[Text]:
		return self._texts

	def draw_onto_frame(self, frame: _Frame, /) -> _Frame:
		frame = self._box.draw_onto_frame(frame)

		##_cv.getFontScaleFromHeight
		##_cv.getTextSize

		# Calculate where to draw the text.
		# `getTextSize` returns ((width, height), baseline)
		details = [
			_cv.getTextSize(
				text=text.label,
				fontFace=FONT,
				fontScale=text.scale,
				thickness=text.thickness,
			)[0]
			for text in self._texts
		]

		if not details:
			return frame

		widths: list[int] = [d[0] for d in details]
		heights: list[int] = [d[1] + self.padding_text_y for d in details]

		total_height: int = sum(heights)
		max_width: int = max(widths)

		# Retrieve the frame's dimensions.
		frame_width: int
		frame_height: int
		frame_height, frame_width = frame.shape[:2]

		# --- Set horizontal position ---
		if max_width > frame_width - (self.padding_frame_x * 2):
			raise ValueError("Text is too wide, unable to handle.")

		max_x: int = frame_width - max_width - self.padding_frame_x
		horizontal_pos: int = max(self.padding_frame_x, min(self._box.left, max_x))

		# --- Set vertical position ---
		# Pin to bottom of the frame as default.
		vertical_pos: int = frame_height - total_height - self.padding_frame_y
		if (
			self._box.bottom + total_height + self.padding_box_y
			<= frame_height - self.padding_frame_y
		):
			# Can go below box.
			vertical_pos = self._box.bottom + self.padding_box_y
		elif self._box.top - total_height - self.padding_box_y >= self.padding_frame_y:
			# Can go above box.
			vertical_pos = self._box.top - total_height - self.padding_box_y
		else:
			# Pin to left side.
			horizontal_pos = self.padding_frame_x

		# --- Drawing ---
		for text, line_height in zip(self._texts, heights):
			vertical_pos += line_height
			frame = text.draw_onto_frame(frame, pos=(horizontal_pos, vertical_pos))

		return frame


class Drawer:
	# Could use sets with tuples
	_faces: list[Face] = []
	_texts: list[Text] = []

	# FIX All of these properties are bad
	@property
	def faces(self) -> list[Face]:
		return self._faces

	@faces.setter
	def faces(self, face: Face | list[Face], /) -> None:
		if isinstance(face, Face):
			face = [face]
		if face in self._faces:
			raise ValueError("Face already exists")
		self._faces += face

	@property
	def texts(self) -> list[Text]:
		return self._texts

	@texts.setter
	def texts(self, text: Text | list[Text], /) -> None:
		if isinstance(text, Text):
			text = [text]
		if text in self._texts:
			raise ValueError("Text already exists")
		self._texts += text

	def draw_onto_frame(self, frame: _Frame, /) -> _Frame:
		for face in self._faces:
			frame = face.draw_onto_frame(frame)

		for text in self._texts:
			frame = text.draw_onto_frame(frame)

		return frame
