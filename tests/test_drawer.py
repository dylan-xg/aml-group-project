
import cv2
import numpy as np

from src.panopticon.drawer import (
	Box, Text, Face, Drawer
)
from src.panopticon.typing import Frame

IMG_SIZE = 500
FIFTH = IMG_SIZE // 5
FOUR_FIFTHS = FIFTH * 4
BASIC_FRAME: Frame = np.ones((IMG_SIZE, IMG_SIZE, 3))
DISPLAY_TIME_MS = 1_000

def show(frame: Frame, /, title: str ='') -> None:
	cv2.imshow(winname=title, mat=frame)
	cv2.waitKey(delay=DISPLAY_TIME_MS)
	cv2.destroyAllWindows()


def create_box() -> Box:
	return Box(left=FIFTH, top=FIFTH, right=FOUR_FIFTHS, bottom=FOUR_FIFTHS)


def create_text(label: str = 'Test', scale: float = 1) -> Text:
	return Text(label=label, position=(FIFTH, FOUR_FIFTHS), scale=scale)


def create_face() -> Face:
	return Face(
		box=create_box(),
		texts=[
			create_text(label='First detail', scale=2),
			create_text(
				label='Second detail longer smaller',
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
	show(result, title=test_face.__name__)


def test_drawer():
	drawer = Drawer()
	drawer.add_face(create_face())
	drawer.add_text(create_text('Inside'))
	result: Frame = drawer.draw_onto_frame(BASIC_FRAME)
	show(result, title=test_drawer.__name__)
