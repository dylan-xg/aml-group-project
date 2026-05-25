
import cv2
import numpy as np

from src.panopticon.drawer import (
	Box, Text, Face, Drawer
)
from src.panopticon.typing import Frame

IMG_SIZE = 100
FIFTH = IMG_SIZE // 5
THREE_FIFTHS = FIFTH * 3
BASIC_FRAME: Frame = np.ones((IMG_SIZE, IMG_SIZE, 3))
DISPLAY_TIME_MS = 1000

def show(frame: Frame, /, title: str ='') -> None:
	cv2.imshow(winname=title, mat=frame)
	cv2.waitKey(delay=DISPLAY_TIME_MS)
	cv2.destroyAllWindows()


def create_box() -> Box:
	return Box(left=FIFTH, top=FIFTH, right=THREE_FIFTHS, bottom=THREE_FIFTHS)


def create_text() -> Text:
	return Text(label='Test', position=(FIFTH, THREE_FIFTHS), scale=1)


def create_face() -> Face:
	return Face(
		box=create_box(),
		details=[
			create_text(),
			Text(
				label='Test2',
				position=(FIFTH, FIFTH),
				scale=0.5
			)
		]
	)


def test_box():
	box = create_box()
	result: Frame = box.draw_onto_frame(BASIC_FRAME)
	show(result, title=test_box.__name__)


def test_text():
	text = create_text()
	result: Frame = text.draw_onto_frame(BASIC_FRAME)
	show(result, title=test_text.__name__)


def test_face():
	face = create_face()
	result: Frame = face.draw_onto_frame(BASIC_FRAME)
	show(result, title=test_text.__name__)


def test_drawer():
	drawer = Drawer()
	drawer.add_face(create_face())
	drawer.add_text(create_text())
	result: Frame = drawer.draw_onto_frame(BASIC_FRAME)
	show(result, title=test_text.__name__)
